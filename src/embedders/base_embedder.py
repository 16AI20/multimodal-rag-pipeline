"""
Base interface class for all document embedders.
Provides common structure and methods for consistent embedder implementation.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ..utils import load_config, get_device


class BaseEmbedder(ABC):
    """Abstract base class for all document embedders."""
    
    def __init__(self, 
                 config_path: str = "conf/config.yaml", 
                 embedding_model: str = None, 
                 embeddings: HuggingFaceEmbeddings = None,
                 document_type: str = None) -> None:
        """
        Initialize the base embedder.
        
        Args:
            config_path: Path to configuration file
            embedding_model: Override for embedding model (optional)
            embeddings: Pre-initialized embeddings instance (optional)
            document_type: Type of document this embedder handles (e.g., 'html', 'pdf')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = load_config(config_path)
        self.device = get_device(self.config)
        self.document_type = document_type
        
        # Get resource directory from config
        if document_type:
            resource_dirs = self.config.get('document_processing', {}).get('resource_dirs', {})
            self.default_resource_dir = resource_dirs.get(document_type, f"corpus/{document_type}")
        else:
            self.default_resource_dir = "corpus"
        
        # Initialize embeddings model
        self._init_embeddings_model(embedding_model, embeddings)
    
    def _init_embeddings_model(self, 
                              embedding_model: str = None, 
                              embeddings: HuggingFaceEmbeddings = None) -> None:
        """Initialize the embeddings model from config or parameters.
        
        Args:
            embedding_model: Override for embedding model name.
            embeddings: Pre-initialized embeddings instance.
        """
        """Initialize the embeddings model from config or parameters."""
        if embeddings is not None:
            self.embeddings = embeddings
            self.logger.info(f"Initialized {self.__class__.__name__} with shared embeddings model")
        else:
            # Get embedding model from config if not provided
            if embedding_model is None:
                embeddings_config = self.config.get('embeddings', {})
                embedding_model = embeddings_config.get('model', 'BAAI/bge-large-en-v1.5')
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': str(self.device)}
            )
            self.logger.info(f"Initialized {self.__class__.__name__} with {embedding_model} on {self.device}")
    
    def get_resource_directory(self) -> str:
        """Get the default resource directory for this embedder type.
        
        Returns:
            Default resource directory path for this embedder type.
        """
        return self.default_resource_dir
    
    @abstractmethod
    def extract_documents(self, resource_dir: str = None) -> List[Document]:
        """
        Extract documents from the specified resource directory.
        
        Args:
            resource_dir: Directory containing documents to process.
                         If None, uses default from config.
        
        Returns:
            List of LangChain Document objects with metadata
        """
        raise NotImplementedError("Subclasses must implement extract_documents")
    
    @abstractmethod
    def embed_to_chroma(self, documents: List[Document], vectorstore: Chroma) -> int:
        """
        Add documents to ChromaDB vectorstore.
        
        Args:
            documents: List of documents to embed
            vectorstore: ChromaDB vectorstore instance
            
        Returns:
            Number of documents successfully embedded
        """
        raise NotImplementedError("Subclasses must implement embed_to_chroma")
    
    def process_and_embed(self, 
                         resource_dir: str = None, 
                         vectorstore: Chroma = None) -> int:
        """
        Complete pipeline: extract documents and embed them.
        
        Args:
            resource_dir: Directory containing documents to process.
                         If None, uses default from config.
            vectorstore: ChromaDB vectorstore instance
            
        Returns:
            Number of documents successfully embedded
        """
        if resource_dir is None:
            resource_dir = self.get_resource_directory()
            
        documents = self.extract_documents(resource_dir)
        
        if vectorstore is None:
            self.logger.error("No vectorstore provided for embedding")
            return 0
            
        return self.embed_to_chroma(documents, vectorstore)
    
    def validate_resource_directory(self, resource_dir: str) -> bool:
        """
        Validate that the resource directory exists and is accessible.
        
        Args:
            resource_dir: Directory path to validate
            
        Returns:
            True if directory exists and is accessible, False otherwise
        """
        path = Path(resource_dir)
        if not path.exists():
            self.logger.warning(f"Resource directory {resource_dir} does not exist")
            return False
        
        if not path.is_dir():
            self.logger.warning(f"Resource path {resource_dir} is not a directory")
            return False
            
        return True
    
    def clean_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to ensure ChromaDB compatibility.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Cleaned metadata dictionary with only supported types
        """
        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned_metadata[key] = value
            else:
                # Convert complex objects to strings
                cleaned_metadata[key] = str(value)
        return cleaned_metadata