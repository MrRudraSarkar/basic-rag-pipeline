from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

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
        "k": 5, # k specifies the number of relevant documents/chunks to retrieve
        "score_threshold": 0.3 # Only return chunks with similarity of 0.3 or higher. We are using cosine similarity which ranges from -1 to 1, where 1 means the vectors are identical, 0 means they are orthogonal (no similarity), and -1 means they are diametrically opposed. So a threshold of 0.3 means we only want chunks that have at least some degree of similarity to the query. You can adjust this threshold based on your needs and the quality of results you want to retrieve.
    }
)

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

# Display results
#print("---- Relevant context ----")

# for i, doc in enumerate(relevant_docs):
#     print(f"Document {i}:\n{doc.page_content}\n")



# Combine the query and the relevant documents
combined_input = f"""Based on the following retrieved documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
# Set up the OllamaLLM object and specify the model
model = OllamaLLM(model="llama3.2:3b", temperature=0.2)

#Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n----- Generated Response -----")
#print("Full result:")
#print(result)
print("Content only:")
print(result)