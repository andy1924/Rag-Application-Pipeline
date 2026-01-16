#This File enables the RAG application to link the context of user query with the previous queries
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

persistDirectory = "db/ChromaDB"

embeddingModel = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory= persistDirectory,
    embedding_function= embeddingModel,
    collection_metadata= {"hnsw:space": "cosine"}
)
model = ChatOpenAI(model="gpt-4o")
chatHistory = []

def askQuestion(question):
    print(f"You asked: {question}")

    if chatHistory:
        messages = [
                       SystemMessage(
                           content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
                   ] + chatHistory + [
                       HumanMessage(content=f"New question: {question}")
                   ]
        result = model.invoke(messages)
        searchQuestion = result.content.strip()
        print(f"Searching Question: {searchQuestion}")
    else:
        searchQuestion = question

    retriever = db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(searchQuestion)

    combinedInput = f"""Based on the following documents, please answer this question: {searchQuestion}
    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}
    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    messages = [
                   SystemMessage(
                       content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
               ] + chatHistory + [
                   HumanMessage(content=combinedInput)
               ]
    result =  model.invoke(messages)
    print(f"AI: {result.content.strip()}")

    chatHistory.append(HumanMessage(content=question))
    chatHistory.append(AIMessage(content=result.content))

def startChat():
    print("-----RAG Based Chat Module Initiated-----\nType 'quit' to exit")
    while True:
        userQuestion = input("\nYour Question: ")

        if userQuestion == "quit":
            break
        else:
            askQuestion(userQuestion)



if __name__ == "__main__":
    startChat()