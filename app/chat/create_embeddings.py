from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.chat.vector_stores.pinecone import vector_store

def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split(text_splitter)

    vector_store.add_documents(docs)

# I ran into this too and found that the pipenv shell and the actual terminal both needed to be exited and reinitialized for the OPENAPI_API_KEY  environment variable to be loaded from the .env file.
