# Import necessary libraries
from doc_loader import load_and_split_documents
from vector import create_or_load_vector_store
from memory import get_conversation_memory
from qa_chain import get_qa_chain
from dotenv import load_dotenv
import os

load_dotenv()

# Function to run CLI
def run_cli():
    docs = load_and_split_documents("sample_pdfs")
    vectordb = create_or_load_vector_store(docs)
    memory = get_conversation_memory()
    chain = get_qa_chain(vectordb, memory)
    print("Gemini-Powered Document Q&A Chatbot (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = chain({"question": query})
        print("\nBot:", result["answer"])
        print("Sources:")
        for doc in result["source_documents"]:
            print(" -", doc.metadata.get("source"))
        print("\n")

if __name__ == "__main__":
    run_cli()