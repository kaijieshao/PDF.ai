import os
import pinecone
from langchain.vectorstores.pinecone import Pinecone
from app.chat.embeddings.openai import embeddings

# Initialize Pinecone client
pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV_NAME")
)

# Create a vector store instance from an existing Pinecone index
vector_store = Pinecone.from_existing_index(
    os.getenv("PINECONE_INDEX_NAME"),             # Get the name of the existing Pinecone index
    embeddings                                    # Pass the embeddings configuration to use
)

# converts the vector store into a retriever object
vector_store.as_retriever()
