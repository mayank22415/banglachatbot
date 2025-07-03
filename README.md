![Screenshot (450)](https://github.com/user-attachments/assets/d20be09e-e645-4a3e-94eb-b39f4c292f73)

Local RAG Voice Chatbot (Bengali)

This project is a fully functional, voice-enabled Retrieval-Augmented Generation (RAG) chatbot that runs entirely on your local machine. It uses a local LLM, local embeddings, and a local vector store to answer questions based on a custom knowledge base you provide. The chatbot is configured to understand queries and respond fluently in Bengali.

<!-- Placeholder for a demo GIF -->

✨ Key Features

100% Local & Private: The entire pipeline (LLM, embeddings, vector database) runs locally. No data is sent to external APIs, ensuring complete privacy.

Retrieval-Augmented Generation (RAG): The chatbot doesn't just rely on its pre-trained knowledge. It retrieves relevant information from your own documents to provide accurate, context-aware answers.

Voice Enabled: Interact with the chatbot using your voice. It features real-time Speech-to-Text (STT) and Text-to-Speech (TTS) for a natural, conversational experience.

Custom Knowledge Base: Easily build a custom knowledge base by dropping your files (.pdf, .csv, .txt, .docx, etc.) into the data directory.

Local LLM: Powered by a quantized Mistral-7B model, running efficiently on your CPU or GPU via CTransformers.

GPU Acceleration: Configurable GPU offloading to significantly speed up response times if you have a compatible NVIDIA GPU.

Dual Interfaces:

Web Interface: A clean, modern chat interface built with Flask and Socket.IO.

Command-Line Interface: A lightweight console version for quick interaction and debugging.

Bengali Focused: The prompt and TTS are pre-configured for high-quality interactions in the Bengali (বাংলা) language.

⚙️ How It Works (Architecture)

The application follows a standard RAG pipeline:

Knowledge Base Indexing:

On startup, the load_knowledge_base function scans the ./data/ directory for supported documents.

It loads and splits these documents into smaller, manageable chunks.

Using Sentence-Transformers, it converts these text chunks into numerical vectors (embeddings).

These embeddings are stored locally in an in-memory FAISS vector index for fast similarity searches.

User Interaction & Response Generation:

The Flask Web App (app.py) or CLI (call.py) captures the user's question (as text or voice).

The question is sent to the LLModel.

The model first queries the FAISS index to find the most relevant document chunks (the "context") related to the question.

The original question, the retrieved context, and a system prompt are combined into a single, comprehensive prompt for the LLM.

The local Mistral-7B model (CTransformers) processes this prompt to generate a helpful answer in Bengali.

The final text response is sent back to the user. For the web app, gTTS also converts this text into speech, which is sent as an audio stream.

📂 Project Structure
Generated code
.
├── local_model/                # Directory for storing the downloaded GGUF model file.
├── data/                       # Add your custom knowledge base files here (PDF, TXT, CSV...).
├── models/
│   └── llm.py                  # Defines the LLModel class, RAG chain, and CTransformers setup.
├── templates/
│   └── chat.html               # The HTML frontend for the web application.
├── app.py                      # Main entry point for the Flask web application.
├── call.py                     # Entry point for the command-line interface chatbot.
├── config.py                   # Central configuration file for the project.
├── faiss.py                    # Contains logic for loading documents and building the FAISS vector store.
├── embedding_generator.py      # (Alternative) For creating and uploading embeddings to Pinecone.
├── requirements.txt            # List of all Python dependencies.
└── README.md                   # This file.

🚀 Setup and Installation
Prerequisites

Python 3.8+

Git

(Optional but Recommended) An NVIDIA GPU with CUDA installed for GPU acceleration.

Step 1: Clone the Repository
Generated bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Step 2: Install Dependencies

It is highly recommended to use a virtual environment.

Generated bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

For GPU Acceleration (NVIDIA):
If you have a compatible NVIDIA GPU and CUDA installed, you can get a significant performance boost. Re-install ctransformers with CUDA support:

Generated bash
pip uninstall ctransformers
pip install ctransformers[cuda]
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

After this, ensure gpu_layers is set to a non-zero value in config.py (e.g., -1 to offload all possible layers).

Step 3: Download the Local LLM

This project is configured to use the mistral-7b-instruct-v0.2.Q4_K_M.gguf model.

Create a directory named local_model:

Generated bash
mkdir local_model
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Download the model file from Hugging Face and place it inside the local_model directory.

Download Link: TheBloke/Mistral-7B-Instruct-v0.2-GGUF

Click the "download" link on the file page to save it.

After downloading, your directory structure should look like this:
./local_model/mistral-7b-instruct-v0.2.Q4_K_M.gguf

Step 4: Add Your Knowledge Base

Create a directory named data:

Generated bash
mkdir data
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Place your custom documents (.pdf, .txt, .csv, .docx, etc.) inside this data directory. The chatbot will use these files to answer questions.

▶️ How to Run
1. Web Application

To start the web-based chat interface:

Generated bash
python app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The application will start indexing your documents (this might take a moment on the first run). Once it's ready, open your web browser and go to: http://127.0.0.1:5000

2. Command-Line Interface

For a simple, terminal-based interaction:

Generated bash
python call.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The script will load the knowledge base and prompt you to start asking questions directly in the console. Type exit to quit.

🔧 Configuration

You can easily modify the chatbot's behavior by editing the config.py file.

Bot_persona: Change the system prompt to alter the bot's personality, rules, or response language.

local_model_path: Update this if you use a different model or file name.

gpu_layers: Set the number of layers to offload to your GPU. -1 offloads all possible layers, while 0 runs on CPU only.

embeddings_model_name: Change the local sentence transformer model for embeddings if needed.

Knowledge_base: The name of the directory containing your knowledge files.

supported_file_types: Add or remove file extensions you want the bot to process.

☁️ Alternative: Using Pinecone (Advanced)

This project also includes embedding_generator.py, which is an alternative script for users who prefer a persistent, cloud-based vector store like Pinecone. This script is not used by the main app.py or call.py flow.

To use it, you would need to:

Install Pinecone: pip install pinecone-client

Set up API keys for Pinecone and (optionally) Mistral AI in a .env file.

Adapt the main application to initialize the vector store from Pinecone instead of FAISS.

License

This project is licensed under the MIT License. See the LICENSE file for details.
