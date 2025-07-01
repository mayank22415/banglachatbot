import os
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import io
import base64
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    JSONLoader
)
from langchain_unstructured import UnstructuredLoader
# NOTE: MistralAIEmbeddings import removed, as we only use local embeddings now
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import get_configurataion
from models.llm import LLModel


def load_knowledge_base(config):
    """
    Loads all documents from the data directory, creates embeddings locally,
    and builds a FAISS vector store in memory.
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
                print(f"  - Skipping unsupported file: {file_name}")
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
        print("--- Knowledge Base is empty. AI will rely on general knowledge. ---")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_to_embed = text_splitter.split_documents(all_docs)
    print(f"\nCreated {len(docs_to_embed)} document chunks for embedding.")

    # Initialize local sentence-transformer embeddings
    print("Using local sentence transformers embeddings...")
    embeddings = SentenceTransformerEmbeddings(
        model_name=config.get("embeddings_model_name", "all-MiniLM-L6-v2")
    )
    
    print("Building FAISS index in memory... (This may take a moment)")
    vectorstore = FAISS.from_documents(docs_to_embed, embeddings)
    
    end_time = time.time()
    print(f"--- Knowledge Base loaded and indexed in {end_time - start_time:.2f} seconds. ---")
    
    return vectorstore


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key-for-socketio'
socketio = SocketIO(app, cors_allowed_origins="*")

print("Initializing application...")
configs = get_configurataion()
# Use the local FAISS loader function
vectorstore = load_knowledge_base(configs) 
# Initialize the AI with the local model and FAISS vectorstore
AI = LLModel(configs, vectorstore) 
print("AI Model Initialized. Application ready.")


def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='bn', slow=False)
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)
        audio_segment = AudioSegment.from_mp3(mp3_buffer)
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        audio_base64 = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')
        return {"success": True, "audio": f"data:audio/wav;base64,{audio_base64}"}
    except Exception as e:
        print(f"TTS Error: {e}")
        return {"success": False, "audio": None}

def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='bn-BD')
        except sr.UnknownValueError:
            text = recognizer.recognize_google(audio, language='en-US')
        os.unlink(temp_path)
        return {"success": True, "text": text}
    except Exception as e:
        print(f"STT Error: {e}")
        return {"success": False, "text": ""}

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/speech-to-text', methods=['POST'])
def handle_speech_to_text():
    audio_blob = request.data
    result = speech_to_text(audio_blob)
    return jsonify(result)

@socketio.on('send_message')
def handle_message(data):
    message = data.get('message', '').strip()
    print(f"Received message: '{message}'")
    if not message:
        return
    response_text = AI.get_response(message)
    print(f"Generated response: '{response_text}'")
    tts_result = text_to_speech(response_text)
    emit('receive_message', {
        'response': response_text,
        'audio': tts_result['audio']
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)