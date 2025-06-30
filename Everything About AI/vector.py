# Import necessary libraries
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

 # Configuring gemini llm api key using .env
load_dotenv()

# Function to create or load vector store from preprocessed docs
def create_or_load_vector_store(docs, persist_dir="vector_store"):
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    index_path = os.path.join(persist_dir, "index.faiss")
    store_path = os.path.join(persist_dir, "index.pkl")

    if os.path.exists(index_path) and os.path.exists(store_path):
        # Load existing FAISS index
        vectordb = FAISS.load_local(persist_dir, embeddings=embedding, allow_dangerous_deserialization=True)
        # Add new documents to the existing index
        vectordb.add_documents(docs)
        vectordb.save_local(persist_dir)
        return vectordb
    else:
        # Create new index
        vectordb = FAISS.from_documents(docs, embedding)
        vectordb.save_local(persist_dir)
        return vectordb