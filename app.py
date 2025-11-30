import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Knowledge AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

class PureGeminiAgent:
    def __init__(self):
        # Try multiple sources for API key
        self.api_key = None
        
        # 1. Try Streamlit secrets first
        try:
            if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
                self.api_key = st.secrets['GOOGLE_API_KEY']
                st.success("✅ API key loaded")
        except Exception as e:
            pass
        
        # 2. Try environment variables as fallback
        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if self.api_key:
                st.success("✅ API key loaded from environment variables")
        
        # 3. If still no key, show error
        if not self.api_key:
            st.error("""
            ❌ GOOGLE_API_KEY not found!
            
            **For Streamlit Cloud:**
            1. Go to your app on share.streamlit.io
            2. Click ••• (3 dots) → Settings
            3. Click "Advanced settings"
            4. Add this exact text:
            
            GOOGLE_API_KEY = "your_actual_api_key_here"
            
            **Make sure to include the quotes!**
            
            **For Local Development:**
            1. Create a .env file
            2. Add: GOOGLE_API_KEY=your_actual_api_key_here
            """)
            st.stop()
        
        # Configure Gemini with the found API key
        genai.configure(api_key=self.api_key)
    
    def load_documents(self, file_paths):
        """Load and process documents without OpenAI dependencies"""
        all_chunks = []
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            try:
                if filename.endswith('.pdf'):
                    text = self.read_pdf(file_path)
                elif filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif filename.endswith(('.docx', '.doc')):
                    text = self.read_docx(file_path)
                else:
                    continue
                
                if text and len(text.strip()) > 0:
                    # Split into chunks
                    chunks = self.split_into_chunks(text, filename)
                    all_chunks.extend(chunks)
                    
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
        
        return all_chunks
    
    def read_pdf(self, file_path):
        """Read PDF files"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PDF reading error: {e}")
            return ""
    
    def read_docx(self, file_path):
        """Read Word documents"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"DOCX reading error: {e}")
            return ""
    
    def split_into_chunks(self, text, filename, chunk_size=1000, chunk_overlap=200):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_text = ' '.join(words[i:i + chunk_size])
            chunks.append({
                'text': chunk_text,
                'source': filename,
                'chunk_id': f"{filename}_chunk_{i}",
                'word_count': len(chunk_text.split())
            })
        
        return chunks
    
    def create_search_index(self, chunks):
        """Create TF-IDF search index"""
        if not chunks:
            return None, None
        
        texts = [chunk['text'] for chunk in chunks]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        return vectorizer, tfidf_matrix
    
    def search_documents(self, query, chunks, vectorizer, tfidf_matrix, top_k=4):
        """Search for relevant document chunks"""
        if vectorizer is None or tfidf_matrix is None:
            return []
        
        # Transform query to TF-IDF vector
        query_vector = vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Similarity threshold
                results.append({
                    'chunk': chunks[idx],
                    'similarity': similarities[idx],
                    'rank': len(results) + 1
                })
        
        return results
    
    def generate_answer(self, question, search_results):
        """Generate answer using Gemini with context from search results"""
        if not self.api_key:
            return "API key not configured. Please check your configuration."
        
        # Prepare context from search results
        context_parts = []
        for result in search_results:
            chunk = result['chunk']
            context_parts.append(f"From {chunk['source']} (relevance: {result['similarity']:.2f}):")
            context_parts.append(chunk['text'])
            context_parts.append("")  # Empty line between chunks
        
        context = "\n".join(context_parts)
        
        try:
            # Use Gemini 2.0 Flash directly
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            You are a knowledgeable AI assistant. Answer the user's question based strictly on the provided context from company documents.

            DOCUMENT CONTEXT:
            {context}

            USER QUESTION: {question}

            IMPORTANT INSTRUCTIONS:
            1. Answer using ONLY the information from the document context above
            2. If the context doesn't contain relevant information, say: "Based on the provided documents, I don't have enough information to answer this question."
            3. Be precise and cite which document the information came from
            4. If multiple documents contain relevant information, synthesize the information
            5. Keep your answer concise but comprehensive
            6. Do not make up information or use external knowledge

            FINAL ANSWER:
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def main():
    st.title("🤖 Knowledge AI Agent")
    st.markdown("Upload documents and get AI-powered answers using Gemini 2.0 Flash!")
    
    # Initialize agent
    agent = PureGeminiAgent()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "docx", "doc"],
            accept_multiple_files=True,
            help="Upload PDF, TXT, DOCX, or DOC files"
        )
        
        if uploaded_files:
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    file_paths.append(file_path)
            
            if file_paths:
                with st.spinner("Processing documents..."):
                    # Load and chunk documents
                    chunks = agent.load_documents(file_paths)
                    
                    if chunks:
                        # Create search index
                        vectorizer, tfidf_matrix = agent.create_search_index(chunks)
                        
                        if vectorizer is not None:
                            st.session_state.document_chunks = chunks
                            st.session_state.vectorizer = vectorizer
                            st.session_state.tfidf_matrix = tfidf_matrix
                            st.session_state.documents_loaded = True
                            st.session_state.agent = agent
                            
                            # Show statistics
                            unique_docs = set(chunk['source'] for chunk in chunks)
                            total_chunks = len(chunks)
                            total_words = sum(chunk['word_count'] for chunk in chunks)
                            
                            st.success(f"✅ Processed {len(unique_docs)} documents")
                            st.info(f"📊 {total_chunks} chunks, {total_words:,} words")
                        else:
                            st.error("❌ Failed to create search index")
                    else:
                        st.error("❌ No documents could be processed")
            
            # Clean up temporary files
            for file_path in file_paths:
                try:
                    os.unlink(file_path)
                except:
                    pass
        
        # Show loaded documents
        if st.session_state.documents_loaded:
            st.subheader("📋 Loaded Documents")
            unique_sources = set(chunk['source'] for chunk in st.session_state.document_chunks)
            for source in unique_sources:
                chunks_from_source = [c for c in st.session_state.document_chunks if c['source'] == source]
                st.write(f"• {source} ({len(chunks_from_source)} chunks)")
        
        # System status
        st.subheader("⚙️ System Status")
        if agent.api_key:
            st.success("✅ Gemini API: Connected")
        else:
            st.error("❌ Gemini API: No API key")
        
        if st.session_state.documents_loaded:
            st.success("✅ Documents: Loaded")
        else:
            st.info("📝 Documents: Not loaded")
        
        # Clear data button
        if st.button("🗑️ Clear All Data"):
            st.session_state.messages = []
            st.session_state.documents_loaded = False
            st.session_state.document_chunks = []
            st.session_state.vectorizer = None
            st.session_state.tfidf_matrix = None
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with Your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Check if documents are loaded
    if not st.session_state.documents_loaded:
        st.info("👋 Welcome to Knowledge AI Agent!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🚀 Get Started")
            st.markdown("""
            1. **Upload documents** in the sidebar (PDF, TXT, DOCX, DOC)
            2. **Ask questions** about your documents
            3. **Get AI-powered answers** with source attribution
            """)
        
        st.stop()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                try:
                    # Search for relevant documents
                    search_results = st.session_state.agent.search_documents(
                        prompt,
                        st.session_state.document_chunks,
                        st.session_state.vectorizer,
                        st.session_state.tfidf_matrix
                    )
                    
                    # Generate answer
                    response = st.session_state.agent.generate_answer(prompt, search_results)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Show sources
                    with st.expander("📚 View Source Documents"):
                        if search_results:
                            st.write(f"**Found {len(search_results)} relevant document chunks:**")
                            for result in search_results:
                                chunk = result['chunk']
                                similarity = result['similarity']
                                content_preview = chunk['text'][:400] + "..." if len(chunk['text']) > 400 else chunk['text']
                                
                                st.markdown(f"**{chunk['source']}** (Relevance: {similarity:.3f})")
                                st.markdown(f"> {content_preview}")
                                st.markdown("---")
                        else:
                            st.info("No relevant documents found for this question.")
                            
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()