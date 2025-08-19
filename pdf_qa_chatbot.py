import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import streamlit as st
from pypdf import PdfReader

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
CHROMA_DB_DIR = 'chroma_db'
MODEL_NAME = 'meta-llama/llama-3-8b-instruct'

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'collection' not in st.session_state:
    st.session_state.collection = None

def load_pdf(uploaded_file):
    """
    Load and extract text from a PDF file
    """
    try:
        pdf_reader = PdfReader(uploaded_file)
        chunks = []
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                chunks.append({
                    'text': text,
                    'page': page_num,
                    'filename': uploaded_file.name
                })
        return chunks
    except Exception as e:
        st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
        return None

def chunk_text(pdf_chunks, chunk_size=1000, overlap=100):
    """
    Split PDF text into smaller chunks with metadata
    """
    all_chunks = []
    for pdf_chunk in pdf_chunks:
        text = pdf_chunk['text']
        
        # Split into smaller chunks
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        # Create chunk objects with metadata
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'metadata': {
                    'filename': pdf_chunk['filename'],
                    'page': pdf_chunk['page'],
                    'chunk_id': idx
                }
            })
    
    return all_chunks

def embed_chunks(chunks, model_name='sentence-transformers/all-mpnet-base-v2'):
    """
    Generate embeddings for text chunks
    """
    model = SentenceTransformer(model_name)
    embedded_chunks = []
    
    for chunk in chunks:
        embedding = model.encode(chunk['text'])
        embedded_chunks.append({
            'embedding': embedding.tolist(),
            'metadata': chunk['metadata'],
            'text': chunk['text']
        })
    
    return embedded_chunks

def store_in_chromadb(embedded_chunks):
    """
    Store embeddings in ChromaDB
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Always start fresh for consistency
    try:
        client.delete_collection("pdf_chunks")
    except:
        pass
        
    collection = client.create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents in batches
    batch_size = 50
    for i in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[i:i + batch_size]
        
        collection.add(
            embeddings=[chunk['embedding'] for chunk in batch],
            metadatas=[chunk['metadata'] for chunk in batch],
            documents=[chunk['text'] for chunk in batch],
            ids=[f"chunk_{i+j}" for j in range(len(batch))]
        )
    
    return collection

def retrieve_relevant_chunks(collection, question, n_results=5):
    """
    Retrieve relevant chunks from ChromaDB
    """
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    question_embedding = model.encode(question).tolist()
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=['metadatas', 'documents']
    )
    
    chunks = []
    if results and 'ids' in results and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            chunks.append({
                'text': results['documents'][0][i],
                'filename': results['metadatas'][0][i]['filename'],
                'page': results['metadatas'][0][i]['page']
            })
    
    return chunks

def ask_llm(context_chunks, question, api_key=OPENROUTER_API_KEY, model=MODEL_NAME):
    """
    Query the LLM with the context and question
    """
    # Format context
    formatted_context = "\n\n".join([
        f"From {chunk['filename']} (Page {chunk['page']}):\n{chunk['text']}"
        for chunk in context_chunks
    ])
    
    prompt = f"""You are an expert at analyzing documents. Answer the following question using ONLY the information provided in the context. 

Question: {question}

Context from documents:
{formatted_context}

Instructions:
1. Format your response in TWO clearly separated sections:
   ANSWER: Provide a clear, direct answer to the question
   REFERENCES: List all source documents and pages used, with brief quotes if relevant

2. For the ANSWER section:
   - Be clear and concise
   - Focus on directly answering the question
   - Do not include citations in this section

3. For the REFERENCES section:
   - List each source used with its document name and page number
   - Include brief relevant quotes if applicable
   - Format as: "Document: [filename] (Page [X]): [relevant quote or context]"

4. If information isn't in the context, clearly state this in the ANSWER section."""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise document analyzer. Always cite sources."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error from API: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="PDF QA Chatbot", page_icon="📚", layout="wide")
st.title("📚 PDF Question Answering Chatbot")

# Define custom styles
custom_css = '''
<style>
    .stTextInput > div > div > input {
        font-size: 1.1em;
    }
    .stButton > button {
        width: 100%;
    }
    .answer-box {
        background-color: var(--secondary-background-color, #f0f2f6);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .references-box {
        border-left: 3px solid var(--primary-color, #4CAF50);
        padding-left: 20px;
        margin: 10px 0;
    }
    .answer-text {
        color: var(--text-color, #31333F);
        font-size: 1.1em;
        line-height: 1.5;
    }
</style>
'''
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            all_embedded_chunks = []
            
            # Process all uploaded files
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    pdf_chunks = load_pdf(uploaded_file)
                    if pdf_chunks:
                        all_chunks = chunk_text(pdf_chunks)
                        embedded_chunks = embed_chunks(all_chunks)
                        all_embedded_chunks.extend(embedded_chunks)
                        st.session_state.processed_files.add(uploaded_file.name)
                        st.success(f"✅ Processed: {uploaded_file.name}")
            
            # Update ChromaDB
            if all_embedded_chunks:
                st.session_state.collection = store_in_chromadb(all_embedded_chunks)
                st.success("✨ All documents indexed and ready!")
    
    st.header("📚 Processed Files")
    if st.session_state.processed_files:
        for file in st.session_state.processed_files:
            st.markdown(f"• {file}")
    else:
        st.info("No files processed yet")

# Main area for Q&A
st.header("❓ Ask Questions")

if not st.session_state.processed_files:
    st.warning("Please upload PDF files first!")
else:
    question = st.text_input("What would you like to know about the documents?")
    
    if st.button("Ask") and question:
        if st.session_state.collection:
            with st.spinner("Searching for answer..."):
                context_chunks = retrieve_relevant_chunks(st.session_state.collection, question)
                
                # Show relevant chunks in expander
                with st.expander("📑 Relevant Context", expanded=False):
                    for chunk in context_chunks:
                        st.markdown(
                            f"**From: {chunk['filename']} (Page {chunk['page']})**\n"
                            f"```\n{chunk['text'][:300]}...\n```"
                        )
                
                # Get and show answer
                answer = ask_llm(context_chunks, question)
                
                # Split answer into sections if formatted correctly
                if "ANSWER:" in answer and "REFERENCES:" in answer:
                    # Split the response into answer and references
                    answer_section = answer.split("REFERENCES:")[0].replace("ANSWER:", "").strip()
                    references_section = "REFERENCES:" + answer.split("REFERENCES:")[1].strip()
                    
                    # Display answer with better formatting
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div class="answer-box"><div class="answer-text">{answer_section}</div></div>', unsafe_allow_html=True)
                    
                    st.markdown("### 📚 References")
                    st.markdown(f'<div class="references-box"><div class="answer-text">{references_section}</div></div>', unsafe_allow_html=True)
                else:
                    # Fallback if the response isn't properly formatted
                    st.markdown("### Answer:")
                    st.markdown(f'<div class="answer-box"><div class="answer-text">{answer}</div></div>', unsafe_allow_html=True)
        else:
            st.error("No document collection found. Please try uploading the documents again.")

# Footer
st.markdown("---")
st.markdown("*Developed By M.Abdulrehman Baig ❤️ using Streamlit*")