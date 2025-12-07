import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize Pinecone
# Ensure your .env has PINECONE_API_KEY
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Check if index exists, if not create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # <--- CHANGED: HuggingFace MiniLM uses 384 dimensions (OpenAI uses 1536)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def ingest_pdf(file_path):
    print(f" Processing {file_path}...")
    
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Create Embeddings (Locally using CPU/GPU)
    print(" Generating Embeddings (this runs locally)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(" Uploading to Pinecone...")
    docsearch = PineconeVectorStore.from_documents(
        documents=splits, 
        embedding=embeddings, 
        index_name=INDEX_NAME
    )
    print("✅ Ingestion Complete!")

if __name__ == "__main__":
    ingest_pdf("test_doc.pdf")