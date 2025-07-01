# app.py

import os
import time
import io
import base64
import tempfile
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import get_configurataion
from models.llm import LLModel

def load_knowledge_base(config):
    """
    Loads documents from the data directory, creates embeddings, and builds a FAISS vector store.
    """
    data_dir = config["Knowledge_base"]
    supported_file_types = config["supported_file_types"]
    print(f"--- Loading Knowledge Base from '{data_dir}' directory ---")
    start_time = time.time()
    
    all_docs = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            ext = file_name.rsplit('.', 1)[-1].lower()

            if ext not in supported_file_types:
                continue

            print(f"  + Processing file: {file_name}")
            loader = None
            try:
                if ext == 'pdf':
                    loader = PyPDFLoader(file_path)
                elif ext == 'csv':
                    loader = CSVLoader(file_path=file_path, autodetect_encoding=True)
                elif ext == 'json':
                    loader = JSONLoader(file_path=file_path, jq_schema='.', text_content=False)
                else:
                    loader = UnstructuredLoader(file_path)
                
                if loader:
                    loaded_docs = loader.load()
                    all_docs.extend(loaded_docs)
            except Exception as e:
                print(f"    ! Error loading {file_name}: {e}")

    if not all_docs:
        print("--- Knowledge Base is empty. AI will rely on its general knowledge only. ---")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_to_embed = text_splitter.split_documents(all_docs)
    print(f"\nCreated {len(docs_to_embed)} document chunks for embedding.")

    print("Initializing local embeddings model...")
    embeddings = SentenceTransformerEmbeddings(
        model_name=config.get("embeddings_model_name", "all-MiniLM-L6-v2")
    )
    
    print("Building FAISS index in memory... (This may take a moment)")
    vectorstore = FAISS.from_documents(docs_to_embed, embeddings)
    
    end_time = time.time()
    print(f"--- Knowledge Base loaded and indexed in {end_time - start_time:.2f} seconds. ---")
    
    return vectorstore


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-very-secret-key-for-socketio'
socketio = SocketIO(app, cors_allowed_origins="*")

print("Initializing application...")
configs = get_configurataion()
vectorstore = load_knowledge_base(configs) 
AI = LLModel(configs, vectorstore) 
print("--- AI Model Initialized. Application is ready to use. ---")


@app.route('/')
def index():
    return render_template('chat.html') # Ensure you have a chat.html template

@socketio.on('send_message')
def handle_message(data):
    user_question = data.get('message', '').strip()
    if not user_question:
        return
        
    print(f"Received query: '{user_question}'")
    
    # Get response from the AI model
    start_time = time.time()
    response_text = AI.get_response(user_question)
    end_time = time.time()
    
    print(f"Generated response: '{response_text}' in {end_time - start_time:.2f}s")

    # Convert response text to speech
    try:
        tts = gTTS(text=response_text, lang='bn', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        audio_base64 = None

    # Emit the complete response back to the client
    emit('receive_message', {
        'response': response_text,
        'audio_base64': audio_base64
    })


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)