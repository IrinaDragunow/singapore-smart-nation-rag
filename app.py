import streamlit as st
import chromadb
import pandas as pd
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import io
import time
import requests
from datetime import datetime
import plotly.express as px
from typing import List, Dict
import re
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Singapore Smart Nation RAG",
    page_icon="🇸🇬",
    layout="wide"
)

# ALLE 5 Singapore Dokumente
SINGAPORE_DOCS = {
    "RIE2025_Research_Innovation_Enterprise": {
        "url": "https://file.go.gov.sg/rie-2025-handbook.pdf",
        "description": "RIE2025 Research Innovation Enterprise Plan",
        "ministry": "Prime Minister's Office"
    },
    "Smart_Nation_2.0_Report": {
        "url": "https://file.go.gov.sg/smartnation2-report.pdf", 
        "description": "Smart Nation 2.0 Strategic Report",
        "ministry": "Smart Nation & Digital Government Office"
    },
    "Budget_2025_Statement": {
        "url": "https://www.mof.gov.sg/docs/librariesprovider3/budget2025/download/pdf/fy2025_budget_statement.pdf",
        "description": "Singapore Budget 2025 Statement", 
        "ministry": "Ministry of Finance"
    },
    "Budget_2025_Booklet": {
        "url": "https://www.mof.gov.sg/docs/librariesprovider3/budget2025/download/pdf/fy2025_budget_booklet_english.pdf",
        "description": "Budget 2025 Citizen Guide",
        "ministry": "Ministry of Finance"
    }
}

# Demo Queries
SINGAPORE_DEMO_QUERIES = [
    "What are the main objectives areas of RIE2025?",
    "What are the key goals of Smart Nation 2.0?",
    "What digital technologies does Singapore plan to invest in?", 
    "How does Singapore plan to improve digital infrastructure?",
    "What are the main research domains in RIE2025?",
    "What digital services are mentioned in Smart Nation 2.0?",
    "How does Singapore plan to support digital transformation?",
    "What are Singapore's AI development strategies?"
]

# Session state
if 'singapore_docs' not in st.session_state:
    st.session_state.singapore_docs = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

class SingaporeRAG:
    def __init__(self):
        self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection("singapore")
        except:
            self.collection = self.client.create_collection("singapore")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def download_and_process_all_docs(self):
        """Download alle 4 Singapore Dokumente"""
        results = {"success": 0, "failed": 0, "total_chunks": 0}
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, (doc_name, doc_info) in enumerate(SINGAPORE_DOCS.items()):
            status.text(f"Processing {doc_info['description']}...")
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(doc_info['url'], headers=headers, timeout=30)
                
                if response.status_code == 200:
                    pdf_file = io.BytesIO(response.content)
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    
                    for page in reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                clean_text = re.sub(r'[^\w\s\.\,\:\;\!\?\-]', ' ', page_text)
                                text += clean_text + " "
                        except:
                            continue
                    
                    if text:
                        chunks = [text[i:i+1500] for i in range(0, len(text), 1200)]
                        embeddings = self.model.encode(chunks)
                        
                        ids = [f"{doc_name}_chunk_{j}" for j in range(len(chunks))]
                        metadatas = [{"document": doc_name, "ministry": doc_info["ministry"]} for _ in chunks]
                        
                        self.collection.add(
                            embeddings=embeddings.tolist(),
                            documents=chunks,
                            metadatas=metadatas,
                            ids=ids
                        )
                        
                        st.session_state.singapore_docs[doc_name] = {
                            "description": doc_info["description"],
                            "ministry": doc_info["ministry"],
                            "chunks": len(chunks)
                        }
                        
                        results["success"] += 1
                        results["total_chunks"] += len(chunks)
                    else:
                        results["failed"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["failed"] += 1
                st.warning(f"Failed to process {doc_name}: {str(e)}")
            
            progress.progress((i + 1) / len(SINGAPORE_DOCS))
        
        status.empty()
        progress.empty()
        return results
    
    def generate_rag_answer(self, query: str, contexts: List[str]) -> str:
        """ECHTE RAG GENERATION mit OpenAI GPT"""
        try:
            if not contexts:
                return "I couldn't find relevant information in the Singapore documents."
            
            # Prepare context
            combined_context = "\n\n".join(contexts[:3])
            
            # RAG Prompt
            prompt = f"""You are an AI assistant specializing in Singapore government policies and initiatives. Based on the following excerpts from official Singapore government documents, please answer the user's question.

Context from Singapore Government Documents:
{combined_context}

User Question: {query}

Instructions:
- Answer based ONLY on the provided context
- Be specific and cite which document/ministry the information comes from
- If the context doesn't contain enough information, say so clearly
- Keep the answer focused and professional
- Use bullet points for lists when appropriate

Answer:"""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert on Singapore government policies and initiatives."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def search_all_docs(self, query: str) -> str:
        """Complete RAG: Retrieval + Generation"""
        try:
            # RETRIEVAL PHASE
            query_embedding = self.model.encode([query])
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            if results['documents'] and results['documents'][0]:
                # Extract relevant contexts
                contexts = []
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0][:3],
                    results['metadatas'][0][:3], 
                    results['distances'][0][:3]
                )):
                    similarity = max(0, 1 - distance)
                    if similarity > 0.05:  # Relevance threshold
                        ministry = metadata.get('ministry', 'Singapore Government')
                        doc_name = metadata.get('document', 'Unknown')
                        context_with_source = f"From {ministry} ({doc_name}): {doc}"
                        contexts.append(context_with_source)
                
                # GENERATION PHASE - OpenAI GPT
                return self.generate_rag_answer(query, contexts)
            else:
                return "No documents available to search."
                
        except Exception as e:
            return f"Search error: {str(e)}"

def main():
    st.title("🇸🇬 Singapore Smart Nation RAG System")
    st.markdown("**Retrieval-Augmented Generation with OpenAI GPT for Singapore Government Documents**")
    
    # API Key Check
    if not openai.api_key:
        st.error("❌ OpenAI API Key not found! Please check your .env file.")
        st.stop()
    else:
        st.success("✅ OpenAI API Key loaded")
    
    # Initialize RAG system
    if st.session_state.vector_db is None:
        st.session_state.vector_db = SingaporeRAG()
    
    rag = st.session_state.vector_db
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Singapore Documents")
        
        for doc_name, doc_info in SINGAPORE_DOCS.items():
            status = "✅" if doc_name in st.session_state.singapore_docs else "⏳"
            st.write(f"{status} {doc_info['description']}")
        
        st.markdown("---")
        
        if st.button("📥 Download All Singapore Documents", type="primary"):
            with st.spinner("Downloading and processing..."):
                results = rag.download_and_process_all_docs()
                st.success(f"✅ Processed {results['success']}/{len(SINGAPORE_DOCS)} documents")
                st.info(f"📄 Total chunks: {results['total_chunks']}")
        
        if st.button("🔄 Reset"):
            st.session_state.singapore_docs = {}
            st.session_state.chat_history = []
            st.session_state.vector_db = SingaporeRAG()
            st.success("Reset complete!")
            st.rerun()
        
        st.write(f"**Status:** {len(st.session_state.singapore_docs)}/{len(SINGAPORE_DOCS)} documents loaded")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Ask About Singapore Policies (Powered by OpenAI)")
        
        st.markdown("**Try these Singapore queries:**")
        for i, query in enumerate(SINGAPORE_DEMO_QUERIES):
            if st.button(f"🔍 {query}", key=f"demo_{i}"):
                if st.session_state.singapore_docs:
                    with st.spinner("🤖 Generating AI answer..."):
                        answer = rag.search_all_docs(query)
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer,
                            "timestamp": datetime.now()
                        })
                        st.write("**AI Answer:**", answer)
                else:
                    st.warning("Please download documents first!")
        
        user_query = st.chat_input("Ask about Singapore Smart Nation policies...")
        
        if user_query:
            if st.session_state.singapore_docs:
                with st.spinner("🤖 AI is analyzing Singapore documents..."):
                    answer = rag.search_all_docs(user_query)
                    st.session_state.chat_history.append({
                        "question": user_query,
                        "answer": answer,
                        "timestamp": datetime.now()
                    })
                    st.write("**Question:**", user_query)
                    st.write("**AI Answer:**", answer)
            else:
                st.warning("Please download documents first!")
    
    with col2:
        st.markdown("### 📊 RAG System Stats")
        
        if st.session_state.singapore_docs:
            st.metric("Documents", f"{len(st.session_state.singapore_docs)}/{len(SINGAPORE_DOCS)}")
            total_chunks = sum(doc["chunks"] for doc in st.session_state.singapore_docs.values())
            st.metric("Text Chunks", total_chunks)
            st.metric("AI Queries", len(st.session_state.chat_history))
            
            st.markdown("**🤖 RAG Pipeline:**")
            st.write("✅ Vector Retrieval")
            st.write("✅ OpenAI Generation")
            st.write("✅ Source Attribution")
        else:
            st.info("No documents loaded yet")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("### 💭 Recent AI Conversations")
        for chat in st.session_state.chat_history[-3:]:
            with st.expander(f"Q: {chat['question'][:50]}..."):
                st.write("**Question:**", chat["question"])
                st.write("**AI Answer:**", chat["answer"])
                st.caption(f"Generated: {chat['timestamp'].strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()