from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings


INDEX_NAME = "document-embeddings"

def run_llm(query: str):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    doc_search = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding_model)

    # Limit to 3 documents during retrieval to reduce token count
    retriever = doc_search.as_retriever(search_kwargs={"k": 2})
    
    # Initialize Groq LLM
    chat = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        verbose=True,
    )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Augmentation with reduced number of documents
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    # Create retrieval chain
    qa = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=stuff_documents_chain
    )
    
    # Get result from LLM
    result = qa.invoke(input={"input": query})
    
    return result  # Return result to print or handle further

if __name__ == "__main__":
    res = run_llm(query="what are the potential names for me based on the documents in the index?")
    print(res["answer"])
