from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Load env variables
load_dotenv()

# Connect to the Vector Store containing the document chunk embeddings. If the DB doesn't exist, run the ingestion pipeline first to create it.
persistent_directory = "db/chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Set up AI model
model = OllamaLLM(model="llama3.2:3b", temperature=0.2)

# Initialize an array to store conversation history and context
chat_history = []

def ask_question(user_question):
    print(f"\n---- You asked: {user_question} ----")

    # Step 1: If chat history exists then make the question clear using context from conversation history
    if chat_history:
        # Ask AI to make reformulate the question uaing conversation history for context
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.strip() # Strip function removes and leading/trailing whitespace characters from the response string
        print(f"Searching for: {search_question}")
    
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents from the vector store using the rewritten question
    retriever = db.as_retriever(search_question={"k": 3}) # k specifies the number of relevant documents/chunks to retrieve. 
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents.")
    for i, doc in enumerate(docs):
        # Show the fist 2 lines of each document
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"    Doc {i}: {preview}...")

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # Step 4: Get the answer 
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type quit to exit.")
    
    while True:
        question = input("\nYour question:")
        
        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()