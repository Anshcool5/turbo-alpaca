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

    # Retrieve relevant documents
    retriever = doc_search.as_retriever(search_kwargs={"k": 2})

    # Initialize Groq LLM
    chat = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        verbose=True,
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a chain for retrieval and document processing
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    qa = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=stuff_documents_chain
    )

    # Get LLM result
    result = qa.invoke(input={"input": query})
    print(result)
    return result["answer"]
