"""HTML content embedder for processing HTML files into vector embeddings.

This module provides functionality to extract structured text from HTML files
using the HTMLParser and convert them into embeddings for semantic search.
"""

from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma

from .base_embedder import BaseEmbedder
from .html_parser import HTMLParser
from ..utils import DocumentChunker


class HTMLEmbedder(BaseEmbedder):
    """Embeds HTML files using existing html_parser.py for content extraction."""
    
    def __init__(self, config_path: str = "conf/config.yaml", embedding_model: str = None, embeddings=None) -> None:
        """Initialize HTML embedder.
        
        Args:
            config_path: Path to configuration file.
            embedding_model: Override for embedding model name.
            embeddings: Pre-initialized embeddings instance.
        """
        # Initialize base class with HTML-specific configuration
        super().__init__(
            config_path=config_path,
            embedding_model=embedding_model,
            embeddings=embeddings,
            document_type="html"
        )
        self.chunker = DocumentChunker(config_path)
    
    def extract_documents(self, html_dir: str = None) -> List[Document]:
        """Extract and convert HTML files to LangChain Documents with citation metadata.
        
        Args:
            html_dir: Directory containing HTML files to process.
            
        Returns:
            List of LangChain Document objects with metadata.
        """
        # Use default directory from config if not provided
        if html_dir is None:
            html_dir = self.get_resource_directory()
            
        self.logger.info(f"Processing HTML files from {html_dir}")
        
        # Validate directory
        if not self.validate_resource_directory(html_dir):
            return []
        
        # Use existing HTML parser
        parser = HTMLParser(html_dir)
        parsed_docs = parser.parse_all_html_files()
        
        # Convert to LangChain Documents with citation metadata
        langchain_docs = []
        for doc in parsed_docs:
            # Ensure we have content
            if not doc.get('content', '').strip():
                self.logger.warning(f"Empty content for {doc.get('file_path', 'unknown')}")
                continue
                
            # Create comprehensive metadata for citations
            metadata = {
                'source': doc.get('source', ''),
                'file_type': 'html',
                'file_path': doc.get('file_path', ''),
                'title': doc.get('title', ''),
                'description': doc.get('description', ''),
                'url': doc.get('url', ''),
                # For citation: primary source identifier
                'citation_source': doc.get('source', doc.get('title', Path(doc.get('file_path', '')).stem))
            }
            
            langchain_doc = Document(
                page_content=doc['content'],
                metadata=metadata
            )
            langchain_docs.append(langchain_doc)
        
        # Apply intelligent chunking
        processed_documents = self.chunker.process_documents(langchain_docs, 'html')
        
        self.logger.info(f"Extracted {len(langchain_docs)} HTML documents into {len(processed_documents)} optimized chunks")
        return processed_documents
    
    def embed_to_chroma(self, documents: List[Document], vectorstore: Chroma) -> int:
        """Add HTML documents to existing ChromaDB vectorstore.
        
        Args:
            documents: List of HTML documents to embed.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
            
        Raises:
            Exception: If embedding process fails.
        """
        if not documents:
            self.logger.warning("No HTML documents to embed")
            return 0
            
        try:
            # Add documents to existing vectorstore
            vectorstore.add_documents(documents)
            self.logger.info(f"Successfully embedded {len(documents)} HTML documents to ChromaDB")
            return len(documents)
            
        except Exception as e:
            self.logger.error(f"Error embedding HTML documents: {e}")
            raise
    
    def process_and_embed(self, html_dir: str = "corpus/html", vectorstore: Chroma = None) -> int:
        """Complete pipeline: extract HTML documents and embed them.
        
        Args:
            html_dir: Directory containing HTML files.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
        """
        documents = self.extract_documents(html_dir)
        
        if vectorstore is None:
            self.logger.error("No vectorstore provided for embedding")
            return 0
            
        return self.embed_to_chroma(documents, vectorstore)