import os
import streamlit as st
import ollama
import time
from rag_qa import process_pdf, retrieval
from config import UPLOAD_FOLDER, CHROMA_DB_PATH

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set page configuration
st.set_page_config(page_title="RAG Q&A with Ollama", layout="wide")
st.title("_RAG Q&A_ with :blue[llama 3.2] :sunglasses:")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection_info" not in st.session_state:
    st.session_state.collection_info = None

if "processing" not in st.session_state:
    st.session_state.processing = False

# Function to handle streaming responses
def stream_response(user_query, collection_info=None):
    if collection_info is None:
        return "Please process a document first."
    
    chroma_db, collection_name = collection_info
    
    # Get relevant documents
    retriever = chroma_db.as_retriever()
    docs = retriever.invoke(user_query)

    if not docs:
        yield "No relevant documents found in the database."
        return

    # Create context from documents
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Generate chat history context
    chat_history = ""
    if len(st.session_state.messages) > 0:
        # Get the last 3 exchanges (6 messages) at most
        recent_messages = st.session_state.messages[-6:] if len(st.session_state.messages) > 6 else st.session_state.messages
        chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages])
    
    # Create a RAG-based prompt
    formatted_prompt = f"""Answer the question based ONLY on the following context. 
    If the information is not in the context, say you don't have enough information to answer accurately.
    
    Context:
    {context}
    
    Previous conversation:
    {chat_history}
    
    Question: {user_query}
    """

    # Generate streaming response using Ollama
    client = ollama.Client(host="http://localhost:11434")
    
    # Use the streaming API
    response_text = ""
    for chunk in client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": formatted_prompt}],
        stream=True
    ):
        if chunk.get("message") and chunk["message"].get("content"):
            content = chunk["message"]["content"]
            response_text += content
            yield content

# Sidebar for PDF upload
st.sidebar.header("Upload a PDF")
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Process PDF if uploaded
if pdf_file:
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.name)
    
    # Save the uploaded file
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    st.sidebar.success(f"File '{pdf_file.name}' uploaded successfully ✅")
    
    # Process the PDF if it hasn't been processed yet
    if st.sidebar.button("Process Document") or st.session_state.processing:
        if not st.session_state.processing:
            st.session_state.processing = True
            st.sidebar.info("Processing document... This may take a minute.")
            
            try:
                chroma_db, collection_name = process_pdf(pdf_path)
                st.session_state.collection_info = (chroma_db, collection_name)
                st.session_state.processing = False
                st.sidebar.success(f"Document processed and stored in ChromaDB collection '{collection_name}'! ✅")
            except Exception as e:
                st.session_state.processing = False
                st.sidebar.error(f"Error processing document: {str(e)}")

# Add a button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Main chat interface
st.markdown("### Chat with your document")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.collection_info is None:
        st.error("Please upload and process a PDF document first!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for response_chunk in stream_response(prompt, st.session_state.collection_info):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)  # Small delay for smoother typing effect
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})