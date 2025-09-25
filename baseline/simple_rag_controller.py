"""
Simple RAG (Retrieval Augmented Generation) Controller
A simplified version of the RAG system for easy use and integration
"""
import sys
sys.path.append("..")

import os
from typing import List, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import chromadb
from docx import Document as DocxDocument

# Import utilities from existing codebase
from settings import config
from utils import convertAllDocToDocx, getAllFilesInDirMatchingFormat
from MetadataAwareChunker import getSectionedChunks, addExtraDocumentWideMetadataForReason


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
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-3-large',
            api_key=self.api_key
        )
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model_name
        )
        
        # Initialize vector store
        self._init_vector_store()
        
        # Set up prompts and chains
        self._setup_chain()
    
    def _init_vector_store(self):
        """Initialize or connect to the Chroma vector store."""
        # Create directory if it doesn't exist
        os.makedirs(self.db_dir_path, exist_ok=True)
        
        # Initialize persistent client
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir_path)
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
    
    def _setup_chain(self):
        """Set up the prompt template and document chain."""
        self.prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context in about 200 words. 
If the answer cannot be found in the context, say so clearly.

<context>
{context}
</context>

Question: {input}

Answer:""")
        
        self.document_prompt = ChatPromptTemplate.from_template("""
Content: {page_content}
Source: {source}
""")
        
        self.doc_chain = create_stuff_documents_chain(
            self.llm, 
            self.prompt_template,
            document_prompt=self.document_prompt
        )
    
    def load_and_chunk_documents(self, 
                                doc_dir_path: str,
                                chunk_size: int = 1000,
                                chunk_overlap: int = 200,
                                file_extensions: List[str] = [".docx", ".txt"]) -> List[Document]:
        """
        Load documents from a directory, chunk them, and return Document objects.
        
        Args:
            doc_dir_path: Path to directory containing documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            file_extensions: List of file extensions to process
            
        Returns:
            List of Document objects
        """
        # Convert any .doc files to .docx
        convertAllDocToDocx(doc_dir_path)
        
        # Get all files matching extensions
        file_list = getAllFilesInDirMatchingFormat(doc_dir_path, file_extensions)
        file_list = [os.path.join(doc_dir_path,file) for file in file_list]
        print(file_list)
        
        all_chunks = getSectionedChunks(file_list,addExtraDocumentWideMetadataForReason)
        
        return all_chunks
    
    def _extract_text_from_docx(self, filepath: str) -> str:
        """Extract text from a docx file."""
        doc = DocxDocument(filepath)
        full_text = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        full_text.append(cell.text)
        
        return '\n'.join(full_text)
    
    def add_documents_to_db(self, 
                           doc_dir_path: str,
                           chunk_size: int = 1000,
                           chunk_overlap: int = 200) -> int:
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
        documents = self.load_and_chunk_documents(
            doc_dir_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if documents:
            # Add to vector store
            batch_size = 100
            i=1
            while (i-1) * batch_size < len(documents):
                print(f" ****** \n\n number of chunks is {len(documents)} and we are on batch {i}. \n\n")
                self.vector_store.add_documents(documents[(i-1)*batch_size:i*batch_size])
                i+=1
            print(f"Added {len(documents)} chunks to the database")
        else:
            print("No documents found to add")
        
        return len(documents)
    
    def clear_database(self):
        """Clear all documents from the database."""
        # Delete the collection
        self.chroma_client.delete_collection(self.collection_name)
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
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(question)
        
        # Generate answer using the chain
        answer = self.doc_chain.invoke({
            "context": retrieved_docs,
            "input": question
        })
        
        return answer, retrieved_docs
    
    def get_document_count(self) -> int:
        """Get the number of documents in the database."""
        # This is an approximation based on collection stats
        collection = self.chroma_client.get_collection(self.collection_name)
        return collection.count()


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
    controller.add_documents_to_db("../all3gppdocsfromrel17and18/Release 17/batch8-2")
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
