import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def loadDocs(docsPath):

    print("------Loading Docs-------")
    if not os.path.exists(docsPath):
        raise FileNotFoundError(f"No file named {docsPath}. Update the File Path to a valid location")

    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(
        path=docsPath,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs= text_loader_kwargs
    )
    documents = loader.load()

    if len(documents) == 0:
        print(f"File location {docsPath} is empty. Add data first.")
    '''for i, doc in enumerate(documents, 0):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")'''

    print("------Docs Loaded-------")
    return documents

def chunkDocs(documents,chunkSize = 1000, chunkOverlap=0):
    print("------Chunking the Document------")

    textSplitter  = CharacterTextSplitter(
        chunk_size=chunkSize,
        chunk_overlap=chunkOverlap
    )
    chunks = textSplitter.split_documents(documents)
    '''if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")'''

    print("-------Chunking Successful-------")
    return chunks

def vectorEmbedding(chunks, persistDirectory ="db/ChromaDB"):
    print("------Starting Vector Embedding-------")
    embeddingModel = OpenAIEmbeddings(model="text-embedding-3-small")

    print("------Creating a Vector DataBase------")
    vectorDB = Chroma.from_documents(
        documents=chunks,
        embedding=embeddingModel,
        persist_directory=persistDirectory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("------Vector DB Created Successfully-------")
    return vectorDB

def main():
    print("Main Ingestion Pipeline")
    #Default Paths
    persistDirectory = "db/ChromaDB"
    Docs = "D:\RagApplication\Docs"

    if os.path.exists(persistDirectory):
        print("Vector DB exists Already.......")
        return

    #Load Documents
    documents = loadDocs(Docs)

    #Chunk the Document
    chunks = chunkDocs(documents)

    #Create a Vector Embeddings and DB
    vectorDB = vectorEmbedding(chunks,persistDirectory)
    print(f"Ingestion Completed. Vector DB is stored at location: {persistDirectory}")

if __name__ == "__main__":
    main()
