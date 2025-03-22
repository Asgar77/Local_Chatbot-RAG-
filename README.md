# RAG Q&A with Ollama and Llama 3.2

A Streamlit-based web application that allows users to upload PDF documents, process them for question-answering, and interact with the documents using Llama 3.2 via Ollama.


![image](https://github.com/user-attachments/assets/137c2acf-8cb6-4214-9173-950b7ef1c4a4)

## Overview

This application implements Retrieval-Augmented Generation (RAG) to enable more accurate and contextual question-answering on your documents. The system:

1. Allows users to upload PDF documents
2. Processes documents by extracting text and creating embeddings
3. Stores document chunks in a vector database (ChromaDB)
4. Retrieves relevant document chunks based on user queries
5. Uses Llama 3.2 through Ollama to generate answers based on the retrieved context

## Features

- 📄 PDF document upload and processing
- 🔍 Semantic search with ChromaDB
- 💬 Interactive chat interface with streaming responses
- 🧠 Contextual responses based on document content
- 📝 Chat history tracking for more coherent conversations
- 🔄 Clear chat history option

## Requirements

- Python 3.8+
- Ollama with Llama 3.2 model installed
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/rag-ollama-qa.git
   cd Local_Chatbot-RAG-
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Ollama by following the instructions at [Ollama's official website](https://ollama.ai/).

5. Pull the required models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text:latest
   ```

## Usage

1. Start the Ollama server:
   ```bash
   ollama serve
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

4. Upload a PDF document using the sidebar

5. Click "Process Document" to extract and embed the document content

6. Ask questions about your document in the chat interface

## Project Structure

- `app.py`: Main Streamlit application file that handles the user interface and interaction
- `rag_qa.py`: Contains the core RAG functionality (document processing, retrieval, and generation)
- `config.py`: Configuration settings for the application

## Configuration

Edit the `config.py` file to modify:
- `UPLOAD_FOLDER`: Directory where uploaded PDFs are stored
- `CHROMA_DB_PATH`: Directory where ChromaDB stores vectors and metadata

## How It Works

### Document Processing
1. When a user uploads a PDF, it's saved to the uploads directory
2. The PDF is processed using UnstructuredPDFLoader to extract text
3. The text is split into chunks using RecursiveCharacterTextSplitter
4. Chunks are embedded using OllamaEmbeddings with the nomic-embed-text model
5. Embeddings and text chunks are stored in ChromaDB

### Question Answering
1. When a user submits a question, the system retrieves relevant chunks from ChromaDB
2. The chunks are combined with the query and recent chat history to create a prompt
3. The prompt is sent to Llama 3.2 via Ollama to generate a response
4. The response is streamed back to the user interface

## Limitations

- The application currently only supports PDF files
- Performance depends on the quality of the Llama 3.2 model and the nomic-embed-text embedding model
- Long documents may take significant time to process
- Requires a machine capable of running Ollama and the Llama 3.2 model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing easy access to Llama 3.2
- [Streamlit](https://streamlit.io/) for the web interface
- [LangChain](https://www.langchain.com/) for the document processing and embedding tools
- [ChromaDB](https://www.trychroma.com/) for the vector database
