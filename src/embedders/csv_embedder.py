"""CSV data embedder for processing CSV files into vector embeddings.

This module provides functionality to extract data from CSV files
and convert them into embeddings for semantic search.
"""

from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma

from .base_embedder import BaseEmbedder
from ..utils import DocumentChunker


class CSVEmbedder(BaseEmbedder):
    """Embeds CSV files using CSVLoader."""
    
    def __init__(self, config_path: str = "conf/config.yaml", embedding_model: str = None, embeddings=None) -> None:
        """Initialize CSV embedder.
        
        Args:
            config_path: Path to configuration file.
            embedding_model: Override for embedding model name.
            embeddings: Pre-initialized embeddings instance.
        """
        # Initialize base class with CSV-specific configuration
        super().__init__(
            config_path=config_path,
            embedding_model=embedding_model,
            embeddings=embeddings,
            document_type="csv"
        )
        self.chunker = DocumentChunker(config_path)
    
    def extract_documents(self, csv_dir: str = None) -> List[Document]:
        """Extract data from CSV files and create LangChain Documents with citation metadata.
        
        Args:
            csv_dir: Directory containing CSV files to process.
            
        Returns:
            List of LangChain Document objects with metadata.
        """
        # Use default directory from config if not provided
        if csv_dir is None:
            csv_dir = self.get_resource_directory()
            
        self.logger.info(f"Processing CSV files from {csv_dir}")
        
        # Validate directory
        if not self.validate_resource_directory(csv_dir):
            return []
        
        csv_path = Path(csv_dir)
        
        documents = []
        csv_files = list(csv_path.glob("*.csv"))
        
        if not csv_files:
            self.logger.warning(f"No CSV files found in {csv_dir}")
            return []
        
        for csv_file in csv_files:
            try:
                self.logger.info(f"Processing CSV: {csv_file.name}")
                
                # Load CSV using CSVLoader
                loader = CSVLoader(
                    file_path=str(csv_file),
                    encoding='utf-8'
                )
                csv_docs = loader.load()
                
                # Process each row from the CSV
                for i, doc in enumerate(csv_docs):
                    # Enhance metadata for citations
                    doc.metadata.update({
                        'source': csv_file.name,
                        'file_type': 'csv',
                        'file_path': str(csv_file),
                        'row_number': i + 1,
                        'total_rows': len(csv_docs),
                        # For citation: use filename without extension
                        'citation_source': f"{csv_file.stem} (Row {i + 1})"
                    })
                    
                    # Only include documents with meaningful content
                    if doc.page_content.strip():
                        documents.append(doc)
                
                self.logger.info(f"Extracted {len(csv_docs)} rows from {csv_file.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing CSV {csv_file}: {e}")
                continue
        
        # Apply content filtering and optimization
        processed_documents = self.chunker.process_documents(documents, 'csv')
        
        self.logger.info(f"Successfully processed {len(documents)} CSV rows into {len(processed_documents)} filtered entries")
        return processed_documents
    
    def embed_to_chroma(self, documents: List[Document], vectorstore: Chroma) -> int:
        """Add CSV documents to existing ChromaDB vectorstore.
        
        Args:
            documents: List of CSV documents to embed.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
            
        Raises:
            Exception: If embedding process fails.
        """
        if not documents:
            self.logger.warning("No CSV documents to embed")
            return 0
            
        try:
            # Process documents in batches to avoid batch size limits
            batch_size = 1000  # Safe batch size for ChromaDB
            total_embedded = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                total_embedded += len(batch)
                self.logger.info(f"Embedded batch {i//batch_size + 1}: {len(batch)} CSV rows")
            
            self.logger.info(f"Successfully embedded {total_embedded} CSV rows to ChromaDB")
            return total_embedded
            
        except Exception as e:
            self.logger.error(f"Error embedding CSV documents: {e}")
            raise
    
    def process_and_embed(self, csv_dir: str = None, vectorstore: Chroma = None) -> int:
        """Complete pipeline: extract CSV documents and embed them.
        
        Args:
            csv_dir: Directory containing CSV files.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
        """
        # Use the base class implementation which handles defaults
        return super().process_and_embed(csv_dir, vectorstore)