import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PDF_PATH = "Gopalswamy_Doraiswamy_Naidu.pdf"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "qwen2.5:1.5b"

if not os.path.exists(PDF_PATH):
    print(f"Error: File '{PDF_PATH}' not found.")
else:
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"FAISS index built with {vectorstore.index.ntotal} vectors")

    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.7)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the question using ONLY "
         "the provided context. If the context doesn't contain the "
         "answer, say 'I don't have enough information to answer this.'"),
        ("human",
         "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain ready!\n")

    question = "Who is G. D. Naidu?"
    answer = rag_chain.invoke(question)
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    print(f"Answer:\n{answer}\n")

    print("--- Sources ---")
    sources = retriever.invoke(question)
    for i, doc in enumerate(sources):
        page_num = doc.metadata.get('page', '?')
        print(f"[{i+1}] (Page {page_num}): {doc.page_content[:150]}...")