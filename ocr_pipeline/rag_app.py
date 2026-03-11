"""
RAG (Retrieval Augmented Generation) App with Streamlit

A document Q&A application that:
1. Accepts PDF uploads
2. Extracts text using OCR pipeline
3. Stores in ChromaDB for semantic search
4. Answers questions using Gemini

Usage:
    streamlit run app.py
"""

import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb import Documents, Embeddings
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import os
import tempfile
import hashlib
from pathlib import Path

from config import get_config
from preprocess import ImagePreprocessor
from text_detector import TextDetector
from gemini_ocr import GeminiOCR
from postprocess import TextPostProcessor
from utils import PDFUtils, ImageUtils, load_document


class DocumentRAG:
    """
    Document Q&A system using ChromaDB and Gemini.

    Stores document chunks in ChromaDB and uses Gemini to answer
    questions based on retrieved context.
    """

    def __init__(self, collection_name: str = "document_qa"):
        """
        Initialize the RAG system.

        Args:
            collection_name: Name of the ChromaDB collection.
        """
        self.config = get_config()
        self.collection_name = collection_name

        # Initialize ChromaDB (in-memory for Streamlit)
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # Reset existing collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass

        # Create collection
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize components
        self.gemini_config = self.config.gemini
        self._setup_gemini()

        # Store document info
        self.document_text = ""
        self.document_name = ""

    def _setup_gemini(self):
        """Setup Gemini API."""
        if not self.gemini_config.api_key:
            self.gemini_config.api_key = os.getenv("GEMINI_API_KEY", "")

        if not self.gemini_config.api_key:
            raise ValueError("Gemini API key not configured")

        genai.configure(api_key=self.gemini_config.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.gemini_config.model_name,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
            }
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for texts using Gemini.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        # Use Gemini for embeddings
        embeddings = []
        for text in texts:
            # Use Gemini's embedContent API
            result = self.model.embed_content(
                model="gemini-embedding-001",
                content=text
            )
            embeddings.append(result.embedding.values)

        return embeddings

    def process_document(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """
        Process a document and store in ChromaDB.

        Args:
            file_path: Path to the document.
            file_name: Name of the file.

        Returns:
            Dictionary with processing results.
        """
        st.info("🔄 Processing document...")

        # Load document
        images = load_document(file_path)

        all_text = []

        # Process each page
        for page_num, image in enumerate(images):
            # Preprocess
            preprocessor = ImagePreprocessor(self.config.preprocessing)
            processed = preprocessor.preprocess(image)

            # Detect text regions
            detector = TextDetector(self.config.text_detector)
            regions = detector.detect(processed)

            # Extract text with Gemini
            ocr = GeminiOCR(self.config.gemini, self.config.ocr)
            ocr_results = ocr.extract_text_from_regions(processed, regions)

            # Collect text
            page_text = "\n".join([r.text for r in ocr_results if r.text.strip()])
            if page_text.strip():
                all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")

        # Combine all text
        self.document_text = "\n\n".join(all_text)
        self.document_name = file_name

        # Split into chunks
        chunks = self._split_into_chunks(self.document_text)

        # Store in ChromaDB
        st.info("💾 Storing in database...")

        # Generate IDs and embeddings
        embeddings = self.get_embeddings(chunks)
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Add to collection
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"source": file_name, "chunk_id": i} for i in range(len(chunks))]
        )

        return {
            "status": "success",
            "pages": len(images),
            "chunks": len(chunks),
            "text_length": len(self.document_text)
        }

    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks for better retrieval.

        Args:
            text: Input text.
            chunk_size: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        # Add last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If chunks are too small, merge them
        if len(chunks) < 3 and len(text) > chunk_size:
            # Split by sentences instead
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""

            for sent in sentences:
                if len(current_chunk) + len(sent) < chunk_size:
                    current_chunk += sent + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        return chunks

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query the document with a question.

        Args:
            question: User's question.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            Dictionary with answer and sources.
        """
        # Get embedding for question
        question_embedding = self.get_embeddings([question])[0]

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )

        # Get relevant chunks
        retrieved_docs = results.get("documents", [[]])[0]
        retrieved_metadatas = results.get("metadatas", [[]])[0]

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information in the document to answer your question.",
                "sources": []
            }

        # Build context from retrieved documents
        context = "\n\n".join([f"Context {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])

        # Generate answer using Gemini
        prompt = f"""You are a helpful assistant answering questions about a document.

Based on the following context extracted from the document, answer the question accurately.
If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

        response = self.model.generate_content(prompt)
        answer = response.text.strip()

        # Format sources
        sources = []
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas)):
            # Get first 200 chars of chunk as preview
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            sources.append({
                "chunk_id": meta.get("chunk_id", i),
                "source": meta.get("source", "unknown"),
                "preview": preview
            })

        return {
            "answer": answer,
            "sources": sources
        }

    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of the stored document."""
        count = self.collection.count()

        return {
            "name": self.document_name,
            "text_length": len(self.document_text),
            "chunks": count
        }


# Streamlit App
def main():
    """Streamlit application."""

    # Page config
    st.set_page_config(
        page_title="Document Q&A",
        page_icon="📄",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: #0e1117;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("📄 Document Q&A with Gemini + ChromaDB")
    st.markdown("Upload a PDF document and ask questions about it!")

    # Sidebar for document upload
    with st.sidebar:
        st.header("📤 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to analyze"
        )

        process_btn = st.button("Process Document", type="primary")

        st.divider()

        # Document info
        st.subheader("📋 Document Info")
        if "doc_info" in st.session_state:
            info = st.session_state.doc_info
            st.write(f"**Name:** {info.get('name', 'N/A')}")
            st.write(f"**Pages:** {info.get('pages', 0)}")
            st.write(f"**Chunks:** {info.get('chunks', 0)}")
            st.write(f"**Text Length:** {info.get('text_length', 0):,} chars")
        else:
            st.info("No document loaded")

        st.divider()

        # Clear document
        if st.button("🗑️ Clear Document"):
            if "rag" in st.session_state:
                del st.session_state["rag"]
            if "doc_info" in st.session_state:
                del st.session_state["doc_info"]
            st.success("Document cleared!")
            st.rerun()

    # Initialize RAG in session state
    if "rag" not in st.session_state:
        st.session_state.rag = None
        st.session_state.doc_info = None

    # Process document
    if uploaded_file is not None and process_btn:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            # Initialize RAG
            rag = DocumentRAG()

            # Process document
            with st.spinner("Processing document..."):
                result = rag.process_document(tmp_path, uploaded_file.name)

            # Store in session
            st.session_state.rag = rag
            st.session_state.doc_info = {
                "name": uploaded_file.name,
                "pages": result["pages"],
                "chunks": result["chunks"],
                "text_length": result["text_length"]
            }

            st.success(f"✅ Document processed successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass

    # Main content - Q&A Section
    st.markdown("---")

    if st.session_state.rag is not None:
        # Question input
        st.subheader("💬 Ask Questions")

        # Example questions
        example_questions = [
            "What is this document about?",
            "What are the main points?",
            "Extract any dates mentioned",
            "Find any names or organizations"
        ]

        col1, col2 = st.columns([3, 1])

        with col1:
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="e.g., What is the total amount?",
                label_visibility="collapsed"
            )

        with col2:
            ask_btn = st.button("Ask", type="primary")

        # Show example questions
        st.markdown("**Quick questions:**")
        cols = st.columns(4)
        for i, q in enumerate(example_questions):
            if cols[i].button(q, key=f"example_{i}"):
                question = q
                ask_btn = True

        # Answer question
        if question and ask_btn:
            with st.spinner("Finding answer..."):
                try:
                    result = st.session_state.rag.query(question)

                    # Display answer
                    st.markdown("### Answer:")
                    st.markdown(f"> {result['answer']}")

                    # Display sources
                    if result["sources"]:
                        st.markdown("### 📚 Sources:")
                        for src in result["sources"]:
                            with st.expander(f"Source {src['chunk_id'] + 1}"):
                                st.markdown(src["preview"])

                except Exception as e:
                    st.error(f"Error answering question: {str(e)}")

    else:
        # Show upload prompt
        st.info("👈 Please upload a PDF document from the sidebar to get started!")

        # Show sample document preview
        st.markdown("### How it works:")
        st.markdown("""
        1. **Upload** - Upload a PDF document using the sidebar
        2. **Process** - Click "Process Document" to extract and index text
        3. **Ask** - Type your question about the document
        4. **Answer** - Get AI-powered answers based on document content
        """)


if __name__ == "__main__":
    main()