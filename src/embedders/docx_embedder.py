"""DOCX document embedder for processing Word documents into vector embeddings.

This module provides functionality to extract text from DOCX files using
UnstructuredWordDocumentLoader and convert them into embeddings for semantic search.
"""

from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_chroma import Chroma

from .base_embedder import BaseEmbedder


class DOCXEmbedder(BaseEmbedder):
    """Embeds DOCX files using UnstructuredWordDocumentLoader."""
    
    def __init__(self, config_path: str = "conf/config.yaml", embedding_model: str = None, embeddings=None) -> None:
        """Initialize DOCX embedder.
        
        Args:
            config_path: Path to configuration file.
            embedding_model: Override for embedding model name.
            embeddings: Pre-initialized embeddings instance.
        """
        # Initialize base class with DOCX-specific configuration
        super().__init__(
            config_path=config_path,
            embedding_model=embedding_model,
            embeddings=embeddings,
            document_type="docx"
        )
    
    def extract_documents(self, docx_dir: str = None) -> List[Document]:
        """Extract text from DOCX files and create LangChain Documents with citation metadata.
        
        Args:
            docx_dir: Directory containing DOCX files to process.
            
        Returns:
            List of LangChain Document objects with metadata.
        """
        # Use default directory from config if not provided
        if docx_dir is None:
            docx_dir = self.get_resource_directory()
            
        self.logger.info(f"Processing DOCX files from {docx_dir}")
        
        # Validate directory
        if not self.validate_resource_directory(docx_dir):
            return []
        
        docx_path = Path(docx_dir)
        
        documents = []
        docx_files = list(docx_path.glob("*.docx"))
        
        if not docx_files:
            self.logger.warning(f"No DOCX files found in {docx_dir}")
            return []
        
        for docx_file in docx_files:
            try:
                self.logger.info(f"Processing DOCX: {docx_file.name}")
                
                # Load DOCX using UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(str(docx_file))
                docx_docs = loader.load()
                
                # Process each section/chunk from the DOCX
                for i, doc in enumerate(docx_docs):
                    # Skip if not a proper Document object
                    if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                        self.logger.warning(f"Skipping invalid document object in {docx_file.name}")
                        continue
                        
                    # Enhance metadata for citations
                    doc.metadata.update({
                        'source': docx_file.name,
                        'file_type': 'docx',
                        'file_path': str(docx_file),
                        'section_number': i + 1,
                        'total_sections': len(docx_docs),
                        # For citation: use filename without extension
                        'citation_source': docx_file.stem
                    })
                    
                    # Clean metadata to ensure ChromaDB compatibility
                    doc.metadata = self.clean_metadata_for_chroma(doc.metadata)
                    
                    # Only include documents with meaningful content
                    if doc.page_content.strip():
                        documents.append(doc)
                
                self.logger.info(f"Extracted {len(docx_docs)} sections from {docx_file.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing DOCX {docx_file}: {e}")
                continue
        
        self.logger.info(f"Successfully processed {len(documents)} DOCX sections total")
        return documents
    
    def embed_to_chroma(self, documents: List[Document], vectorstore: Chroma) -> int:
        """Add DOCX documents to existing ChromaDB vectorstore.
        
        Args:
            documents: List of DOCX documents to embed.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
            
        Raises:
            Exception: If embedding process fails.
        """
        if not documents:
            self.logger.warning("No DOCX documents to embed")
            return 0
            
        try:
            # Add documents to existing vectorstore
            vectorstore.add_documents(documents)
            self.logger.info(f"Successfully embedded {len(documents)} DOCX sections to ChromaDB")
            return len(documents)
            
        except Exception as e:
            self.logger.error(f"Error embedding DOCX documents: {e}")
            raise
    
    def process_and_embed(self, docx_dir: str = None, vectorstore: Chroma = None) -> int:
        """Complete pipeline: extract DOCX documents and embed them.
        
        Args:
            docx_dir: Directory containing DOCX files.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
        """
        # Use the base class implementation which handles defaults
        return super().process_and_embed(docx_dir, vectorstore)