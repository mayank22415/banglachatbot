import time
from config import get_configurataion
from models.llm import LLModel
from faiss import load_knowledge_base
def launchbot(configs):
    
    vectorstore = load_knowledge_base(configs)

    if vectorstore:
        print("FAISS index created with local documents.")
    else:
        print("No documents found, proceeding with general knowledge.")
    
    AI = LLModel(configs, vectorstore)
    return AI
       
if __name__ == "__main__":
    configs = get_configurataion()
    print("--- Launching Command-Line Bot (Local LLM) ---")
    AI = launchbot(configs)
    print("\n--- Bot is ready. Type 'exit' to quit. ---")

    while True:
        question = input("এখানে আপনার প্রশ্ন লিখুন: ")
        if question == "exit":
            break
        start_time = time.time()
        response = AI.get_response(question)
        
        print("\n Time taken: ", round(time.time() - start_time, 3), 'sec  \n')
        print("\033[92m" + response + "\033[0m")
