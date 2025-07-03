
import os
from dotenv import load_dotenv

load_dotenv()

def get_configurataion():

    config = {
        "Bot_persona": [
            """You are a helpful and polite AI assistant for a chatbot. You must follow these rules strictly:
            1. Your final response MUST be in Bengali (বাংলা) language ONLY.
            2. Your entire response must be UNDER 60 words. Keep it short, natural, and to the point.
            3. Use the provided "Context" to answer the user's question. 
            4. If the "Context" is not relevant or does not contain the answer, use your general knowledge to answer the user's question, but still follow all other rules.
            5. Never state that you are an AI or refer to the context directly."""
        ],
      
        "local_model_path": os.path.join("local_model", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "gpu_layers": -1, 
        "embeddings_model_name": "all-MiniLM-L6-v2",
        "Knowledge_base": "data",
        "supported_file_types": ['c', 'cpp', 'csv', 'docx', 'html', 'java', 'json', 'md', 'pdf', 'php', 'pptx', 'py', 'rb', 'tex', 'txt', 'css', 'jpeg', 'jpg', 'js', 'gif', 'png', 'tar', 'ts', 'xlsx', 'xml', 'zip'],
    }
    return config
