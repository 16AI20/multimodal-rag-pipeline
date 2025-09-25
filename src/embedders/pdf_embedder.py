"""PDF document embedder for processing PDF files into vector embeddings.

This module provides functionality to extract text from PDF files using
UnstructuredPDFLoader and convert them into embeddings for semantic search.
Includes intelligent heading detection and enhanced citation formatting.
"""

from pathlib import Path
from typing import List, Dict, Any
import re
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_chroma import Chroma

from .base_embedder import BaseEmbedder
from ..utils import DocumentChunker


class PDFEmbedder(BaseEmbedder):
    """Embeds PDF files using UnstructuredPDFLoader."""
    
    def __init__(self, config_path: str = "conf/config.yaml", embedding_model: str = None, embeddings=None) -> None:
        """Initialize PDF embedder.
        
        Args:
            config_path: Path to configuration file.
            embedding_model: Override for embedding model name.
            embeddings: Pre-initialized embeddings instance.
        """
        # Initialize base class with PDF-specific configuration
        super().__init__(
            config_path=config_path,
            embedding_model=embedding_model,
            embeddings=embeddings,
            document_type="pdf"
        )
        self.chunker = DocumentChunker(config_path)
    
    def _extract_headings_and_structure(self, content: str) -> Dict[str, Any]:
        """Extract headings and structural elements from PDF content.
        
        Args:
            content: Full text content of the PDF.
            
        Returns:
            Dictionary containing extracted headings and structure information.
        """
        structure_info = {
            'headings': [],
            'sections': [],
            'has_toc': False
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for heading patterns (all caps, numbered sections, etc.)
            if re.match(r'^[A-Z\s]{3,}$', line) and len(line) < 100:
                structure_info['headings'].append(line)
                current_section = line
            elif re.match(r'^\d+\.?\s+[A-Z]', line) and len(line) < 100:
                structure_info['headings'].append(line)
                current_section = line
            elif re.match(r'^(Chapter|Section|Part)\s+\d+', line, re.IGNORECASE):
                structure_info['headings'].append(line)
                current_section = line
            
            # Check for table of contents indicators
            if re.search(r'table\s+of\s+contents|contents', line, re.IGNORECASE):
                structure_info['has_toc'] = True
        
        return structure_info
    
    def _create_enhanced_citation(self, doc_name: str, page_num: int, heading: str = None) -> str:
        """Create enhanced citation format with optional heading context.
        
        Args:
            doc_name: Name of the document.
            page_num: Page number.
            heading: Optional heading/section name.
            
        Returns:
            Formatted citation string.
        """
        base_citation = f"{doc_name} (Page {page_num})"
        
        if heading and len(heading) < 50:  # Only include short, meaningful headings
            return f"{doc_name} (Page {page_num}, {heading})"
        
        return base_citation
    
    def extract_documents(self, pdf_dir: str = None) -> List[Document]:
        """Extract text from PDF files and create LangChain Documents with enhanced citation metadata.
        
        Args:
            pdf_dir: Directory containing PDF files to process.
            
        Returns:
            List of LangChain Document objects with metadata.
        """
        # Use default directory from config if not provided
        if pdf_dir is None:
            pdf_dir = self.get_resource_directory()
            
        self.logger.info(f"Processing PDF files from {pdf_dir}")
        
        # Validate directory
        if not self.validate_resource_directory(pdf_dir):
            return []
        
        pdf_path = Path(pdf_dir)
        
        documents = []
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {pdf_dir}")
            return []
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing PDF: {pdf_file.name}")
                
                # Load PDF using UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                
                if not pdf_docs:
                    self.logger.warning(f"No documents extracted from {pdf_file.name}")
                    continue
                
                # Extract document structure for all pages
                all_content = " ".join([doc.page_content for doc in pdf_docs])
                doc_structure = self._extract_headings_and_structure(all_content)
                
                doc_name = pdf_file.stem
                
                # Process each page/chunk from the PDF
                for i, doc in enumerate(pdf_docs):
                    try:
                        # Skip if not a proper Document object
                        if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                            self.logger.warning(f"Skipping invalid document object in {pdf_file.name}")
                            continue
                            
                        page_num = i + 1
                        
                        # Try to identify current section/heading for this page
                        current_heading = None
                        page_content = doc.page_content.strip()
                        
                        # Look for headings in the current page content
                        page_lines = page_content.split('\n')[:5]  # Check first few lines
                        for line in page_lines:
                            line = line.strip()
                            if line in doc_structure['headings']:
                                current_heading = line
                                break
                            # Also check for heading patterns in this page
                            elif re.match(r'^[A-Z\s]{3,}$', line) and len(line) < 80:
                                current_heading = line
                                break
                        
                        # Create enhanced citation
                        citation_source = self._create_enhanced_citation(
                            doc_name, page_num, current_heading
                        )
                        
                        # Enhanced metadata for citations
                        enhanced_metadata = {
                            'source': pdf_file.name,
                            'file_type': 'pdf',
                            'file_path': str(pdf_file),
                            'page_number': page_num,
                            'total_pages': len(pdf_docs),
                            'document_name': doc_name,
                            'current_heading': current_heading,
                            'has_structure': len(doc_structure['headings']) > 0,
                            'has_toc': doc_structure['has_toc'],
                            # Enhanced citation format: "Document Name (Page X)" or "Document Name (Page X, Heading)"
                            'citation_source': citation_source
                        }
                        
                        # Update document metadata
                        doc.metadata.update(enhanced_metadata)
                        
                        # Clean metadata to ensure ChromaDB compatibility
                        doc.metadata = self.clean_metadata_for_chroma(doc.metadata)
                        
                        # Only include documents with meaningful content
                        if page_content:
                            documents.append(doc)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing document {i} in {pdf_file.name}: {e}")
                        continue
                
                self.logger.info(f"Extracted {len(pdf_docs)} pages from {pdf_file.name}")
                self.logger.info(f"Found {len(doc_structure['headings'])} headings in {pdf_file.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing PDF {pdf_file}: {e}")
                continue
        
        # Apply intelligent chunking and content enhancement
        processed_documents = self.chunker.process_documents(documents, 'pdf')
        
        self.logger.info(f"Successfully processed {len(documents)} PDF pages into {len(processed_documents)} optimized chunks")
        return processed_documents
    
    def embed_to_chroma(self, documents: List[Document], vectorstore: Chroma) -> int:
        """Add PDF documents to existing ChromaDB vectorstore.
        
        Args:
            documents: List of PDF documents to embed.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
            
        Raises:
            Exception: If embedding process fails.
        """
        if not documents:
            self.logger.warning("No PDF documents to embed")
            return 0
            
        try:
            # Add documents to existing vectorstore
            vectorstore.add_documents(documents)
            self.logger.info(f"Successfully embedded {len(documents)} PDF pages to ChromaDB")
            return len(documents)
            
        except Exception as e:
            self.logger.error(f"Error embedding PDF documents: {e}")
            raise
    
    def process_and_embed(self, pdf_dir: str = None, vectorstore: Chroma = None) -> int:
        """Complete pipeline: extract PDF documents and embed them.
        
        Args:
            pdf_dir: Directory containing PDF files.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
        """
        # Use the base class implementation which handles defaults
        return super().process_and_embed(pdf_dir, vectorstore)