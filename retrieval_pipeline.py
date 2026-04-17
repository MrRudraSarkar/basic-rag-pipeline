from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

#Use when we are using cloud models which need API keys for authentication. Since we are running ollama locally we don't need authentication now
#from dotenv import load_dotenv
#load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_metadata = {"hnsw:space": "cosine"}
)

# Search for relevant documents

query = "Which island does SpaceX lease for it's launches in the Pacific?"
#query = ""

#retriever = db.as_retriever(search_kwargs={"k": 3})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3 # Only return chunks with cosine similarity of 0.3 or higher
    }
)

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

# Display results
print("---- Relevant context ----")

for i, doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")