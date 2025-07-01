import os
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredFileLoader,
    JSONLoader
)
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class EmbeddingGenerator:
    """
    Handles the creation and management of document embeddings using Mistral AI or local embeddings.
    """
    def __init__(self, configs):
        self.configs = configs
        
        # Initialize embeddings based on provider choice
        embeddings_provider = configs.get("embeddings_provider", "mistral")
        
        if embeddings_provider == "mistral":
            # Use Mistral embeddings
            self.embeddings_model = MistralAIEmbeddings(
                model=configs["embeddings_model_name"],
                mistral_api_key=configs["mistral_api_key"]
            )
        elif embeddings_provider == "sentence_transformers":
            # Use local sentence transformers (free alternative)
            self.embeddings_model = SentenceTransformerEmbeddings(
                model_name=configs.get("embeddings_model_name", "all-MiniLM-L6-v2")
            )
        else:
            raise ValueError(f"Unsupported embeddings provider: {embeddings_provider}")

        # Initialize Pinecone client
        self.pc = PineconeClient(api_key=configs["pinecone_api_key"])
        self.index_name = configs["pinecone_index_name"]
        self.knowledge_base_dir = configs["Knowledge_base"]
        self.supported_file_types = configs["supported_file_types"]

    def get_index_stats(self):
        """Returns stats for the current Pinecone index using the new client."""
        if self.index_name in self.pc.list_indexes().names():
            index = self.pc.Index(self.index_name)
            return index.describe_index_stats()
        return f"Index '{self.index_name}' does not exist."

    def create_embeddings(self):
        """
        Processes files, creates embeddings, and uploads them to Pinecone.
        Skips uploading if the index is not empty.
        """
        print(f"Loading documents from '{self.knowledge_base_dir}'...")

        stats = self.get_index_stats()
        if isinstance(stats, dict) and stats.get('total_vector_count', 0) > 0:
            print("Pinecone index is not empty. Skipping embedding creation.")
            return Pinecone.from_existing_index(self.index_name, self.embeddings_model)

        all_docs = []
        for root, _, files in os.walk(self.knowledge_base_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                ext = "." + file_name.rsplit('.', 1)[-1].lower()

                if file_name.rsplit('.', 1)[-1].lower() not in self.supported_file_types:
                    print(f"  - Skipping unsupported file: {file_name}")
                    continue

                print(f"  + Processing file: {file_name}")
                loader = None
                try:
                    if ext == '.pdf':
                        loader = PyPDFLoader(file_path)
                    elif ext == '.csv':
                        loader = CSVLoader(file_path=file_path, autodetect_encoding=True)
                    elif ext == '.json':
                        loader = JSONLoader(file_path=file_path, jq_schema='.', text_content=False)
                    else:
                        loader = UnstructuredFileLoader(file_path)
                    
                    if loader:
                        all_docs.extend(loader.load())
                except Exception as e:
                    print(f"    ! Error loading {file_name}: {e}")
        
        if not all_docs:
            print("No new documents to process.")
            pinecone_index = self.pc.Index(self.index_name)
            return Pinecone(index=pinecone_index, embedding=self.embeddings_model, text_key="text")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_to_embed = text_splitter.split_documents(all_docs)
        print(f"Created {len(docs_to_embed)} chunks for embedding.")

        print(f"\nUploading {len(docs_to_embed)} chunks to Pinecone index '{self.index_name}'...")
        vectorstore = Pinecone.from_documents(
            documents=docs_to_embed,
            embedding=self.embeddings_model,
            index_name=self.index_name
        )
        print("Embedding creation and upload complete!")

        return vectorstore