from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub

def run_llm(query: str, INDEX_NAME):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    doc_search = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding_model)

    # Initialize Groq LLM
    chat = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        verbose=True,
    )

    #retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a chain for retrieval and document processing
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    qa = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=stuff_documents_chain
    )

    # Get LLM result
    result = qa.invoke(input={"input": query})
    print(result)
    return result["answer"]

def handle_generic(user_query):
    """Call the LLM again to generate a conversational response."""
    
    # Define a conversational prompt
    chat_prompt = ChatPromptTemplate.from_template("""
    You are a friendly chatbot that engages in casual conversations with users.
    Respond naturally to the user's message.

    User: {user_query}
    AI:
    """)

    # Format prompt with user input
    formatted_prompt = chat_prompt.format(user_query=user_query)
    
    # Call the LLM again
    chat = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
    response = chat.invoke(formatted_prompt)

    return response.strip()  # Return a clean response