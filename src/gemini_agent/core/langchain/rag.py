import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)

class LangChainRAG:
    def __init__(
        self,
        api_key: str,
        persist_directory: str = "./chroma_db",
        model_name: str = "models/embedding-001",
    ):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name, google_api_key=api_key
        )
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def load_and_index_directory(self, directory_path: str, glob: str = "**/*.py"):
        """
        Loads documents from a directory and indexes them in the vector store.
        """
        loader = DirectoryLoader(
            directory_path, glob=glob, loader_cls=TextLoader, show_progress=True
        )
        documents = loader.load()
        self.index_documents(documents)

    def index_documents(self, documents: List[Document]):
        """
        Splits and indexes documents.
        """
        texts = self.text_splitter.split_documents(documents)
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
        else:
            self.vectorstore.add_documents(texts)

    def query(self, query_text: str, k: int = 5) -> List[Document]:
        """
        Queries the vector store for relevant documents.
        """
        if self.vectorstore is None:
            # Try to load from disk
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
            else:
                return []
        
        return self.vectorstore.similarity_search(query_text, k=k)
