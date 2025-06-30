# Import necessary libraries
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Function to get the qa_chain for chatbot
def get_qa_chain(vectordb, memory):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt = PromptTemplate(
        input_variables=["summaries", "question"],
        template="""
You are an AI assistant. Use the following summaries/docs to answer the question.
Be concise and accurate. If you are unsure about the answer, say "I don't know".

Summaries:
{summaries}

Question:
{question}

Answer:
"""
    )
    question_generator = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("Rephrase the question based on the chat: {question}")
    )
    combine_docs_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=prompt
    )
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=True,
        output_key="answer",  # Fixes memory bug
        verbose=True
    )
    return chain