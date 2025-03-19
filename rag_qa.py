import os
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from config import CHROMA_DB_PATH

# Set up embedding model - using a model with consistent dimensions
EMBEDDING_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "llama3.2"

def process_pdf(pdf_path):
    """Processes the PDF, extracts text, and stores embeddings in ChromaDB."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"File path '{pdf_path}' does not exist.")

    # Load PDF
    pdf_loader = UnstructuredPDFLoader(pdf_path)
    pdf_data = pdf_loader.load()
    print("PDF data loaded...")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    data_chunks = text_splitter.split_documents(pdf_data)
    print("Data chunks created...")

    # Create a fresh collection each time to avoid dimension mismatch
    collection_name = f"rag_documents_{os.path.basename(pdf_path).replace('.','_')}"
    
    # Initialize the embedding function
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL, show_progress=True)
    
    # Store in ChromaDB with a new collection
    chroma_db = Chroma.from_documents(
        data_chunks,
        embedding_function,
        persist_directory=CHROMA_DB_PATH,
        collection_name=collection_name,
    )

    print(f"Data stored in ChromaDB collection '{collection_name}' ✅")
    return chroma_db, collection_name

def retrieval(user_query, collection_info=None):
    """Retrieves relevant documents and generates a response using Ollama."""
    if collection_info is None:
        # This would be used if you want to load an existing collection
        raise ValueError("No collection information provided")
    
    chroma_db, collection_name = collection_info if isinstance(collection_info, tuple) else (collection_info, "rag_documents")
    
    retriever = chroma_db.as_retriever()
    docs = retriever.invoke(user_query)

    if not docs:
        return "No relevant documents found in the database."

    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Create a RAG-based prompt
    formatted_prompt = f"""Answer the question based ONLY on the following context:
    {context}
    Question: {user_query}
    """

    # Generate response using Ollama
    client = ollama.Client(host="http://localhost:11434")
    response = client.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": formatted_prompt}])

    return response["message"]["content"]