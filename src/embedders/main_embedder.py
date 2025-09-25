#!/usr/bin/env python
"""
Main embedding orchestrator script that processes all file types in the corpus
and creates a unified ChromaDB collection for RAG applications.

Usage:
    python -m src.embedders.main_embedder
    python -m src.embedders.main_embedder --corpus-dir custom_corpus --db-path custom_db
"""

import os
# Set environment variables before importing other libraries
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import our custom embedders
from .html_embedder import HTMLEmbedder
from .pdf_embedder import PDFEmbedder
from .docx_embedder import DOCXEmbedder
from .csv_embedder import CSVEmbedder
from .audio_embedder import AudioEmbedder
from .image_embedder import ImageEmbedder

# Import utilities
from ..utils import load_config, get_device, setup_logging


class MainEmbedder:
    """Main orchestrator for embedding all corpus file types into ChromaDB."""
    
    def __init__(self, 
                 corpus_dir: str = "corpus",
                 db_path: str = "vector_store",
                 config_path: str = "conf/config.yaml",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 collection_name: str = "rag_collection") -> None:
        """Initialize the main embedder orchestrator.
        
        Args:
            corpus_dir: Directory containing all corpus files.
            db_path: Path for ChromaDB storage.
            config_path: Path to configuration file.
            embedding_model: HuggingFace embedding model name.
            collection_name: ChromaDB collection name.
        """
        
        # Setup logging first
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = load_config(config_path)
        self.device = get_device(self.config)
        self.corpus_dir = Path(corpus_dir)
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize embeddings model (shared across all embedders)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': str(self.device)}
        )
        
        # Initialize ChromaDB vectorstore
        self.vectorstore = None
        self._initialize_vectorstore()
        
        # Initialize all embedders
        self.embedders = self._initialize_embedders(embedding_model)
        
        self.logger.info(f"Initialized MainEmbedder with {embedding_model} on {self.device}")
        self.logger.info(f"Corpus directory: {self.corpus_dir}")
        self.logger.info(f"Database path: {self.db_path}")
    
    def _initialize_vectorstore(self) -> None:
        """Initialize or connect to existing ChromaDB vectorstore.
        
        Raises:
            Exception: If vectorstore initialization fails.
        """
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.db_path
            )
            self.logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vectorstore: {e}")
            raise
    
    def _initialize_embedders(self, embedding_model: str) -> Dict[str, Any]:
        """Initialize all embedder classes with shared embeddings model.
        
        Args:
            embedding_model: HuggingFace embedding model name.
            
        Returns:
            Dictionary mapping file types to embedder instances.
        """
        return {
            'html': HTMLEmbedder(embeddings=self.embeddings),
            'pdf': PDFEmbedder(embeddings=self.embeddings),
            'docx': DOCXEmbedder(embeddings=self.embeddings),
            'csv': CSVEmbedder(embeddings=self.embeddings),
            'audio': AudioEmbedder(embeddings=self.embeddings),
            'image': ImageEmbedder(embeddings=self.embeddings)
        }
    
    def process_html_files(self) -> int:
        """Process all HTML files in corpus/html/.
        
        Returns:
            Number of HTML documents successfully embedded.
        """
        html_dir = self.corpus_dir / "html"
        if not html_dir.exists():
            self.logger.warning(f"HTML directory {html_dir} does not exist")
            return 0
        
        self.logger.info("Starting HTML file processing...")
        count = self.embedders['html'].process_and_embed(str(html_dir), self.vectorstore)
        self.logger.info(f"Completed HTML processing: {count} documents embedded")
        return count
    
    def process_pdf_files(self) -> int:
        """Process all PDF files in corpus/pdf/.
        
        Returns:
            Number of PDF documents successfully embedded.
        """
        pdf_dir = self.corpus_dir / "pdf"
        if not pdf_dir.exists():
            self.logger.warning(f"PDF directory {pdf_dir} does not exist")
            return 0
        
        self.logger.info("Starting PDF file processing...")
        count = self.embedders['pdf'].process_and_embed(str(pdf_dir), self.vectorstore)
        self.logger.info(f"Completed PDF processing: {count} documents embedded")
        return count
    
    def process_docx_files(self) -> int:
        """Process all DOCX files in corpus/docx/.
        
        Returns:
            Number of DOCX documents successfully embedded.
        """
        docx_dir = self.corpus_dir / "docx"
        if not docx_dir.exists():
            self.logger.warning(f"DOCX directory {docx_dir} does not exist")
            return 0
        
        self.logger.info("Starting DOCX file processing...")
        count = self.embedders['docx'].process_and_embed(str(docx_dir), self.vectorstore)
        self.logger.info(f"Completed DOCX processing: {count} documents embedded")
        return count
    
    def process_csv_files(self) -> int:
        """Process all CSV files in corpus/csv/.
        
        Returns:
            Number of CSV rows successfully embedded.
        """
        csv_dir = self.corpus_dir / "csv"
        if not csv_dir.exists():
            self.logger.warning(f"CSV directory {csv_dir} does not exist")
            return 0
        
        self.logger.info("Starting CSV file processing...")
        count = self.embedders['csv'].process_and_embed(str(csv_dir), self.vectorstore)
        self.logger.info(f"Completed CSV processing: {count} documents embedded")
        return count
    
    def process_audio_files(self) -> int:
        """Process all audio files in corpus/audio/.
        
        Returns:
            Number of audio segments successfully embedded.
        """
        audio_dir = self.corpus_dir / "audio"
        if not audio_dir.exists():
            self.logger.warning(f"Audio directory {audio_dir} does not exist")
            return 0
        
        self.logger.info("Starting audio file processing...")
        count = self.embedders['audio'].process_and_embed(str(audio_dir), self.vectorstore)
        self.logger.info(f"Completed audio processing: {count} documents embedded")
        return count
    
    def process_image_files(self) -> int:
        """Process all image files in corpus/images/.
        
        Returns:
            Number of image documents successfully embedded.
        """
        image_dir = self.corpus_dir / "images"
        if not image_dir.exists():
            self.logger.warning(f"Image directory {image_dir} does not exist")
            return 0
        
        self.logger.info("Starting image file processing...")
        count = self.embedders['image'].process_and_embed(str(image_dir), self.vectorstore)
        self.logger.info(f"Completed image processing: {count} documents embedded")
        return count
    
    def process_all_files(self) -> Dict[str, int]:
        """Process all file types in the corpus directory.
        
        Returns:
            Dictionary mapping file types to number of documents embedded.
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING COMPLETE CORPUS EMBEDDING PROCESS")
        self.logger.info("=" * 60)
        
        results = {}
        total_documents = 0
        
        # Process each file type
        file_types = [
            ('HTML', self.process_html_files),
            ('PDF', self.process_pdf_files),
            ('DOCX', self.process_docx_files),
            ('CSV', self.process_csv_files),
            ('Audio', self.process_audio_files),
            ('Image', self.process_image_files)
        ]
        
        for file_type, process_func in file_types:
            try:
                count = process_func()
                results[file_type.lower()] = count
                total_documents += count
                
            except Exception as e:
                self.logger.error(f"Error processing {file_type} files: {e}")
                results[file_type.lower()] = 0
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("EMBEDDING PROCESS COMPLETE")
        self.logger.info("=" * 60)
        
        for file_type, count in results.items():
            self.logger.info(f"{file_type.upper()}: {count} documents embedded")
        
        self.logger.info(f"TOTAL: {total_documents} documents embedded to ChromaDB")
        self.logger.info(f"Collection: {self.collection_name}")
        self.logger.info(f"Database path: {self.db_path}")
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current ChromaDB collection.
        
        Returns:
            Dictionary containing collection statistics and metadata.
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get sample of metadata to show file types
            if count > 0:
                results = collection.get(limit=min(count, 100))
                file_types = set()
                sources = set()
                
                for metadata in results.get('metadatas', []):
                    if metadata:
                        file_types.add(metadata.get('file_type', 'unknown'))
                        sources.add(metadata.get('citation_source', 'unknown'))
                
                return {
                    'total_documents': count,
                    'file_types': list(file_types),
                    'sample_sources': list(sources)[:10],  # First 10 sources
                    'collection_name': self.collection_name,
                    'database_path': self.db_path
                }
            else:
                return {
                    'total_documents': 0,
                    'file_types': [],
                    'sample_sources': [],
                    'collection_name': self.collection_name,
                    'database_path': self.db_path
                }
                
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {'error': str(e)}


def main() -> None:
    """CLI entry point for the main embedder."""
    parser = argparse.ArgumentParser(description="Embed all corpus files into ChromaDB for RAG")
    parser.add_argument("--corpus-dir", default="corpus", help="Path to corpus directory")
    parser.add_argument("--db-path", default="vector_store", help="Path to ChromaDB storage")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5", help="HuggingFace embedding model")
    parser.add_argument("--collection-name", default="rag_collection", help="ChromaDB collection name")
    parser.add_argument("--info-only", action="store_true", help="Only show collection info, don't process files")
    
    args = parser.parse_args()
    
    try:
        # Initialize embedder
        embedder = MainEmbedder(
            corpus_dir=args.corpus_dir,
            db_path=args.db_path,
            embedding_model=args.embedding_model,
            collection_name=args.collection_name
        )
        
        if args.info_only:
            # Just show collection info
            info = embedder.get_collection_info()
            logger.info("\nChromaDB Collection Information:")
            logger.info("=" * 40)
            for key, value in info.items():
                logger.info(f"{key}: {value}")
        else:
            # Process all files
            embedder.process_all_files()
            
            # Show final info
            logger.info("\nFinal Collection Information:")
            info = embedder.get_collection_info()
            for key, value in info.items():
                logger.info(f"{key}: {value}")
            
    except KeyboardInterrupt:
        logger.info("\\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()