"""
Simple RAG (Retrieval Augmented Generation) Controller
A simplified version of the RAG system for easy use and integration
"""
import sys

from DBClient import DBClient
sys.path.append("..")

import os
from typing import List, Tuple, Optional
from RAGQAEngine import RAGQAEngine
from docx import Document as DocxDocument

# Import utilities from existing codebase
from settings import config
from utils import convertAllDocToDocx, getAllFilesInDirMatchingFormat, getTokenCount, Document


CROSS_CONTEXT_BENCHMARK_COLL_NAME = "pure_specs"
EVOLUTION_BENCHMARK_COLL_NAME = "specs_and_discussions"

class SimpleRAGController:
    """
    A simplified RAG controller that handles document ingestion, 
    retrieval, and question answering using Langchain and Chroma.
    """
    
    def __init__(self, 
                 db_dir_path: str = "./chroma_db",
                 collection_name: str = CROSS_CONTEXT_BENCHMARK_COLL_NAME,
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-4o-mini"):
        """
        Initialize the RAG controller.
        
        Args:
            db_dir_path: Path to store the Chroma database
            collection_name: Name of the collection in Chroma
            api_key: OpenAI API key (uses config if not provided)
            model_name: Name of the OpenAI model to use
        """
        self.db_dir_path = db_dir_path
        self.collection_name = collection_name
        self.api_key = api_key or config.get("API_KEY")
        self.model_name = model_name
        
        
        self.qa_engine = RAGQAEngine(prompt_template_file_path="../prompt.txt",
                                     model_name=self.model_name, api_key=self.api_key)

        # Initialize vector store
        self._init_vector_store()
            
    def _init_vector_store(self):
        """Initialize or connect to the Chroma vector store."""
        # Create directory if it doesn't exist
        os.makedirs(self.db_dir_path, exist_ok=True)
        
        # Initialize persistent client
        self.db_client = DBClient(collection_name=self.collection_name, db_dir_path=self.db_dir_path)
    
    def add_documents_to_db(self, 
                           doc_dir_path: str) -> int:
        """
        Load documents from a directory and add them to the vector database.
        
        Args:
            doc_dir_path: Path to directory containing documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of documents added
        """
        # Load and chunk documents
        convertAllDocToDocx(doc_dir_path)
        
        # Get all files matching extensions
        file_list = getAllFilesInDirMatchingFormat(doc_dir_path, file_extensions=[".docx"])
        file_list = [os.path.join(doc_dir_path,file) for file in file_list]
        self.db_client.updateDBFromFileList(file_list,doc_dir=doc_dir_path)

        return len(file_list)

    def clear_database(self):
        """Clear all documents from the database."""
        # Delete the collection
        self.db_client.delFromDB(filter={})
        # Reinitialize
        self._init_vector_store()
        print("Database cleared")
    
    def runController(self, 
                     question: str,
                     k: int = 4) -> Tuple[str, List[Document]]:
        """
        Answer a question using RAG.
        
        Args:
            question: The question to answer
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (answer, retrieved_documents)
        """
        # Retrieve relevant documents
        retrieved_docs = self.db_client.queryDB(query_text=question, k=k)

        # Generate answer using the chain
        answer = self.qa_engine.get_answer_from_context(question, retrieved_docs)
        
        return answer, retrieved_docs


# Convenience functions for quick usage
def create_rag_controller(db_path: str = "./chroma_db", 
                         collection_name: str = "simple_rag") -> SimpleRAGController:
    """Create a RAG controller with default settings."""
    return SimpleRAGController(
        db_dir_path=db_path,
        collection_name=collection_name
    )


def quick_setup_and_query(doc_dir: str, 
                         question: str,
                         db_path: str = "./chroma_db") -> Tuple[str, List[Document]]:
    """
    Quick function to set up a RAG system, load documents, and answer a question.
    
    Args:
        doc_dir: Directory containing documents to load
        question: Question to answer
        db_path: Path to store the database
        
    Returns:
        Tuple of (answer, retrieved_documents)
    """
    # Create controller
    controller = create_rag_controller(db_path)
    
    # Add documents
    controller.add_documents_to_db(doc_dir)
    
    # Answer question
    return controller.runController(question)


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    controller = SimpleRAGController(
        db_dir_path="./db",
        collection_name=EVOLUTION_BENCHMARK_COLL_NAME
    )
    
    # Add documents from a directory
    #controller.add_documents_to_db("./data")
    controller.add_documents_to_db("../all3gppdocsfromrel17and18/Release 18/batch11")
    """root_folder = "../all3gppdocsfromrel17and18/Release 17"

    for subdir,dirs,files in os.walk(root_folder):
        print(f"****Adding docs from subdir {subdir} ****\n")
        controller.add_documents_to_db(subdir)
        print(f"**** finished adding docs from {subdir} **** \n")
    """
    # Ask a question
    question = "What was the motivation behind introducing the feature or updates reflected in this change history, and how do they improve the overall 3GPP standard?"
    answer, docs = controller.runController(question,k=5)
    print(f"Answer: {answer}")
    #print(f"\nRetrieved {docs} documents")
    
    # Example 2: Quick setup and query
    # answer, docs = quick_setup_and_query(
    #     doc_dir="./data",
    #     question="What is 5G?",
    #     db_path="./quick_rag_db"
    # )
