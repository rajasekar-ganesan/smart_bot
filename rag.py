from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

FAISS_PATH = "G:/smart/faiss_index"

def create_or_load_vectorstore(pdf_path):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("PDF PATH:", pdf_path)
    print("EXISTS:", os.path.exists(pdf_path))

    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF not found: {pdf_path}")

    if os.path.exists(FAISS_PATH):
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("Creating new FAISS index...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(FAISS_PATH)

    return vectorstore