# pip install "unstructured[md]"
# pip install unstructured
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
import pickle
import os
from dotenv import load_dotenv
import time

load_dotenv()

embedding_function = OpenAIEmbeddings()

character_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

db = Chroma(
    embedding_function=embedding_function,
    persist_directory="./11-Langchain-Bot/langchain_documents_db/",
)


def read_documentation():
    new_memory_load = pickle.loads(
        open("/Users/jacob/src/consulting/MyRareData/langchainbot_with_sources/11-Langchain-Bot/langchain_documents.pkl", "rb").read()
    )
    # print(new_memory_load)

    docs = character_text_splitter.split_documents(new_memory_load)
    for doc in docs:
        db.add_documents([doc])
        time.sleep(0.001)
        db.persist()
        print("+")



def main():
    read_documentation()


if __name__ == "__main__":
    main()
