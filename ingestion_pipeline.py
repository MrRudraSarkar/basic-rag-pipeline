import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="doc"):
    """Load all documents from the specified directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory path {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the doc directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"} # Make sure you specify this. Explanation below 
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add relevant documents to the directory.")
    
    """for i, doc in enumerate(documents[:2]):
        print(f"Document {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" Metadata: {doc.metadata}")"""

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    
    """Split documents into smaller chunks with overlap for better processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    """if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f" Source : {chunk.metadata["source"]}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"content:")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks.")"""

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """\nCreate and persist ChromaDB"""
    print("Creating embeddings and storing in CHromaDB...")
    
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("---- Finished creating vector store ----")

    print(f"Vector store created and saved to {persist_directory}.")
    return vectorstore

def main():
    print("Starting the ingestion pipeline...")

    #1. Load the files
    documents = load_documents(docs_path="docs")

    #2. Chunking the files
    chunks = split_documents(documents)

    #3. Embedding and storing into the Vector db
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()





# """Important Note:
# So far i think one of the well-explained rag tutorials, for those of you wondering about free models, you can set up OLlama on your local computer. 
# I am using M1 Pro 16 gigs of ram and it is way too fast. 


# Setup Ollama by yourself and run below commands
# ollama pull llama3.2:3b        # Small, fast, good for testing
# ollama pull nomic-embed-text   # For embeddings (very light) 

# So whenever Harish calls 
# llm = ChatOpenAI(model="gpt-3.5-turbo")
# embeddings = OpenAIEmbeddings()


# Just replace it with 
# from langchain_ollama import OllamaLLM, OllamaEmbeddings

# llm = OllamaLLM(model="llama3.2:1b")
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# loader_kwargs={"encoding": "utf-8"} need explanation: 
#
# The problem was UnicodeDecodeError as in the TextLoader uses windows default cp1252 encoding. 
# So to fix this we explicitly specify the encoding as utf-8 which is more universal and can handle a wider range of characters.
