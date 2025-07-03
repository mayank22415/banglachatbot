
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

class LLModel:
    def __init__(self, configs, vectorstore):
        self.configs = configs
        self.vectorstore = vectorstore
        
        persona_template = self.configs.get("Bot_persona", ["You are a helpful assistant."])[0]
        self.prompt = PromptTemplate(
            template=f"""{persona_template}
            
            Context: {{context}}
            Question: {{question}}
            
            Helpful Answer:""",
            input_variables=["context", "question"]
        )

        self.llm = CTransformers(
            model=self.configs["local_model_path"],
            model_type='mistral',
            config={
                'max_new_tokens': 150, 
                'temperature': 0.7,
                'gpu_layers': self.configs.get("gpu_layers", 0)
            }
        )
        print(f"--- CTransformers LLM loaded with {self.configs.get('gpu_layers', 0)} GPU layers. ---")
        if self.vectorstore:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type='stuff', 
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 3} 
                ),
                return_source_documents=False,
                chain_type_kwargs={'prompt': self.prompt}
            )
        else:
            self.qa_chain = None

    def get_response(self, question):
        """
        Takes a user's question, retrieves relevant context, and generates a response.
        """
        try:
            if self.qa_chain:
                result = self.qa_chain({"query": question})
                return result.get("result", "দুঃখিত, আমি এই মুহূর্তে উত্তর দিতে পারছি না।").strip()
            else:
                prompt = self.prompt.format(context="No context provided.", question=question)
                response = self.llm.invoke(prompt)
                return response.strip()
                
        except Exception as e:
            print(f"An error occurred in LLModel: {e}")
            return "একটি ত্রুটি ঘটেছে। অনুগ্রহ করে আবার চেষ্টা করুন।"
