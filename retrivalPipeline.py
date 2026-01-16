from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistDirectory = "db/ChromaDB"

embeddingModel = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory= persistDirectory,
    embedding_function= embeddingModel,
    collection_metadata= {"hnsw:space": "cosine"}
)
query = "How much did Microsoft pay to acquire GitHub"
retriever = db.as_retriever(search_kwargs = {"k":3})
relevantDocs = retriever.invoke(query)

print(f"Query: {query}")
print("----Start of Result----")
#for i, doc in enumerate(relevantDocs,0):
#   print(f"Document {i}: \n{doc.page_content}\n")
#print("----End of Result----")

combinedInput = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevantDocs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
model = ChatOpenAI(model='gpt-4o')
messages = [
    SystemMessage(content='You are a helpful assistance'),
    HumanMessage(content = combinedInput)
]
result = model.invoke(messages)
print(result.content)
'''
Other Test Question Sample
1. "What was NVIDIA's first graphics accelerator called?"
2. "Which company did NVIDIA acquire to enter the mobile processor market?"
3. "What was Microsoft's first hardware product release?"
4. "How much did Microsoft pay to acquire GitHub?"
5. "In what year did Tesla begin production of the Roadster?"
6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
8. "What was the original name of Microsoft before it became Microsoft?"
9.
'''