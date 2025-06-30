# Import necessary libraries for ui
import streamlit as st
from doc_loader import load_and_split_documents
from vector import create_or_load_vector_store
from memory import get_conversation_memory
from qa_chain import get_qa_chain
import io

# Set UI page configuaration
st.set_page_config(page_title="Q&A  Chatbot", layout="wide")
st.title("Gemini-Powered Document Q&A Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize LangChain chain
if "chain" not in st.session_state:
    with st.spinner("Loading documents and building vector DB..."):
        docs = load_and_split_documents("sample_pdfs")  # Make sure your PDFs are here
        vectordb = create_or_load_vector_store(docs)
        memory = get_conversation_memory()
        st.session_state.chain = get_qa_chain(vectordb, memory)

# User's Question input
user_query = st.text_input("Ask me a question about your uploaded document:")

# Handles user query
if user_query:
    with st.spinner("Generating answer..."):
        result = st.session_state.chain({"question": user_query})
        answer = result["answer"]
        sources = set(doc.metadata.get("source", "Unknown") for doc in result["source_documents"])

        # Display result
        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        for src in sources:
            st.write("-", src)

        # Save to session history
        st.session_state.chat_history.append({
            "question": user_query,
            "answer": answer,
            "sources": list(sources)
        })

# Show conversation history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Conversation History")
    for i, entry in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {entry['question']}")
        st.markdown(f"**A{i+1}:** {entry['answer']}")
        if entry["sources"]:
            st.markdown(f"**Sources:** {', '.join(entry['sources'])}")
        st.markdown("---")

# 📥 Export conversation history
def export_history():
    output = io.StringIO()
    for i, entry in enumerate(st.session_state.chat_history):
        output.write(f"Q{i+1}: {entry['question']}\n")
        output.write(f"A{i+1}: {entry['answer']}\n")
        if entry["sources"]:
            output.write(f"Sources: {', '.join(entry['sources'])}\n")
        output.write("-" * 50 + "\n")
    return output.getvalue().encode("utf-8")

if st.session_state.chat_history:
    st.download_button(
        label="Download Conversation History",
        data=export_history(),
        file_name="chat_history.txt",
        mime="text/plain"
    )