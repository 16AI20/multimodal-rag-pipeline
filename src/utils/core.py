"""Core utilities module for the RAG system.

This module provides essential utilities for configuration management, device setup,
logging configuration, reproducibility controls, and document processing optimizations.
It handles environment setup, PyTorch device detection (MPS/CUDA/CPU), and provides
intelligent document chunking for improved embedding quality across different file types.

Key Components:
    - Configuration loading with Hydra and fallback mechanisms
    - Device detection and MPS sparse tensor compatibility
    - Global seed setting for reproducibility
    - Logging setup with YAML configuration support
    - Document chunking utilities for cross-modal retrieval optimization

Typical usage:
    from src.utils.core import load_config, get_device, setup_logging
    
    config = load_config()
    device = get_device(config)
    setup_logging()
"""

import os
import re
import random
import logging
import logging.config
from typing import Optional, List

# Third-party imports
import torch
import yaml
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Disable telemetry and prevent multiprocessing issues before importing other packages
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Module logger
logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = "conf", config_name: str = "config") -> DictConfig:
    """Load configuration using Hydra with robust fallback mechanisms.
    
    Attempts to load configuration using Hydra's compose mechanism. If that fails,
    falls back to direct YAML loading. Returns an empty configuration as a last resort.
    
    Args:
        config_path: Path to config directory or file. If ends with '.yaml',
                    treats as file path. Otherwise treats as directory path.
                    Defaults to "conf".
        config_name: Name of config file without extension. Defaults to "config".
    
    Returns:
        DictConfig: Hydra configuration object containing the loaded configuration.
                   Returns empty config if all loading attempts fail.
    
    Raises:
        No exceptions are raised - all errors are handled gracefully with fallbacks.
    
    Example:
        >>> config = load_config("conf", "config")
        >>> device_pref = config.get("device", "cpu")
    """
    logger.info(f"Loading configuration from path: {config_path}, name: {config_name}")
    
    try:
        # Handle None config_path
        if config_path is None:
            config_path = "conf"
            logger.debug("Config path was None, using default 'conf'")
            
        # Handle both directory paths and file paths for backward compatibility
        if config_path.endswith('.yaml'):
            # If a full file path is provided, extract directory and filename
            config_dir = os.path.dirname(config_path)
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            logger.debug(f"Extracted config_dir: {config_dir}, config_name: {config_name} from file path")
        else:
            # Use as directory path
            config_dir = config_path
        
        # Convert relative path to absolute path
        if not os.path.isabs(config_dir):
            config_dir = os.path.abspath(config_dir)
            logger.debug(f"Converted to absolute path: {config_dir}")
        
        # Initialize Hydra with config directory
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
            logger.info(f"Successfully loaded Hydra config from {config_dir}/{config_name}.yaml")
            return cfg
            
    except Exception as e:
        # Ensure variables are defined for error handling
        if 'config_dir' not in locals():
            config_dir = config_path or "conf"
        if 'config_name' not in locals():
            config_name = "config"
        
        logger.warning(f"Failed to load Hydra config from '{config_dir}/{config_name}.yaml'. Using fallback.")
        logger.debug(f"Hydra loading error: {e}")
        
        # Fallback to basic YAML loading
        fallback_path = os.path.join(config_dir, f"{config_name}.yaml")
        if os.path.exists(fallback_path):
            try:
                with open(fallback_path, "r", encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f)
                    logger.info(f"Successfully loaded fallback YAML config from {fallback_path}")
                    return OmegaConf.create(config_dict)
            except yaml.YAMLError as yaml_e:
                logger.warning(f"Failed to parse fallback config: {yaml_e}")
        else:
            logger.warning(f"Fallback config file not found: {fallback_path}")
        
        # Return empty config as last resort
        logger.error("All config loading attempts failed, returning empty configuration")
        return OmegaConf.create({})


def get_device(config: DictConfig) -> torch.device:
    """Detect and configure the optimal PyTorch device based on availability and configuration.
    
    Determines the best available device considering hardware capabilities, configuration
    preferences, and environment constraints. Handles special cases for containerized
    environments and MPS compatibility issues on Apple Silicon.
    
    Args:
        config: Hydra configuration object containing device preferences.
               Expected to have a 'device' key with values: 'mps', 'cuda', 'cpu', 'auto'.
    
    Returns:
        torch.device: Configured PyTorch device object ready for tensor operations.
    
    Example:
        >>> config = load_config()
        >>> device = get_device(config)
        >>> tensor = torch.randn(3, 3).to(device)
    """
    logger.info("Detecting optimal PyTorch device")
    
    # Force CPU in containerized environments
    if os.getenv("FORCE_CPU_DEVICE", "false").lower() == "true":
        logger.info("Container environment detected - forcing CPU device")
        return torch.device("cpu")
    
    pref = config.get("device", "mps")
    logger.debug(f"Device preference from config: {pref}")

    if pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device
    elif pref == "mps" and torch.backends.mps.is_available():
        try:
            # Test MPS availability
            test_tensor = torch.tensor([1.0], device='mps')
            logger.debug("MPS device test successful")
            # Enable MPS with sparse tensor fixes
            _setup_mps_sparse_tensor_fixes()
            logger.info("Using MPS with sparse tensor compatibility fixes")
            return torch.device("mps")
        except RuntimeError as e:
            logger.warning(f"MPS detected but not functional: {e}. Falling back to CPU")
            return torch.device("cpu")
    elif pref == "cpu":
        logger.info("Using CPU device as specified in config")
        return torch.device("cpu")
    elif pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Auto-detected CPU device")
        return device
    else:
        logger.warning(f"Unknown device preference '{pref}', defaulting to CPU")
        return torch.device("cpu")


def _setup_mps_sparse_tensor_fixes() -> None:
    """Set up compatibility fixes for MPS sparse tensor operations.
    
    Apple's Metal Performance Shaders (MPS) backend doesn't fully support sparse tensors.
    This function applies monkey patches to handle sparse tensor operations gracefully
    by converting them to dense tensors before MPS operations.
    
    Environment variables set:
        - PYTORCH_ENABLE_MPS_FALLBACK: Enables CPU fallback for unsupported MPS ops
        - TOKENIZERS_PARALLELISM: Disabled to prevent multiprocessing warnings
    
    Patches applied:
        - torch.Tensor.to: Converts sparse tensors to dense before MPS transfer
        - torch.nn.Embedding.forward: Ensures embedding weights are dense on MPS
    
    Raises:
        No exceptions are raised - patches are applied best-effort with logging.
    """
    logger.debug("Setting up MPS sparse tensor compatibility fixes")
    
    # Enable PyTorch MPS fallback for unsupported operations
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    
    # Disable tokenizers parallelism to avoid fork warnings
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    # Monkey patch common tensor operations to handle sparse tensors on MPS
    original_to = torch.Tensor.to
    
    def patched_to(self, *args, **kwargs):
        """Patched tensor.to() method that handles sparse tensors on MPS."""
        # If moving sparse tensor to MPS, convert to dense first
        if self.is_sparse and len(args) > 0:
            device_arg = args[0]
            if isinstance(device_arg, torch.device) and device_arg.type == 'mps':
                # Convert to dense, then move to MPS
                dense_tensor = self.to_dense()
                logger.debug("Converted sparse tensor to dense for MPS transfer")
                return original_to(dense_tensor, *args, **kwargs)
            elif isinstance(device_arg, str) and 'mps' in device_arg:
                # Handle string device specification
                dense_tensor = self.to_dense()
                logger.debug("Converted sparse tensor to dense for MPS transfer (string device)")
                return original_to(dense_tensor, *args, **kwargs)
        
        # For non-sparse tensors or non-MPS moves, use original method
        return original_to(self, *args, **kwargs)
    
    # Apply the patch
    torch.Tensor.to = patched_to
    logger.debug("Applied torch.Tensor.to patch for MPS sparse tensor compatibility")
    
    # Also patch embedding layers which commonly use sparse tensors
    try:
        import torch.nn as nn
        original_embedding_forward = nn.Embedding.forward
        
        def patched_embedding_forward(self, input_tensor):
            """Patched embedding forward that ensures dense weights on MPS."""
            # Ensure embedding weight is dense on MPS
            if self.weight.device.type == 'mps' and self.weight.is_sparse:
                self.weight.data = self.weight.data.to_dense()
                logger.debug("Converted sparse embedding weights to dense for MPS")
            return original_embedding_forward(self, input_tensor)
        
        nn.Embedding.forward = patched_embedding_forward
        logger.debug("Applied torch.nn.Embedding.forward patch for MPS compatibility")
        
    except Exception as e:
        logger.warning(f"Could not patch embedding layers: {e}")


def safe_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Safely move tensor to device with sparse tensor handling for MPS.
    
    Provides a safe way to move tensors to different devices, particularly handling
    the case where sparse tensors need to be moved to MPS devices (which don't
    support sparse operations natively).
    
    Args:
        tensor: PyTorch tensor to move to the target device.
        device: Target PyTorch device (cpu, cuda, mps).
    
    Returns:
        torch.Tensor: Tensor moved to the target device. Sparse tensors are
                     converted to dense format when moving to MPS devices.
    
    Example:
        >>> sparse_tensor = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1, 2])
        >>> mps_device = torch.device('mps')
        >>> dense_tensor = safe_tensor_to_device(sparse_tensor, mps_device)
    """
    if tensor.is_sparse and device.type == 'mps':
        # Convert sparse tensor to dense before moving to MPS
        logger.debug(f"Converting sparse tensor to dense for MPS device transfer")
        return tensor.to_dense().to(device)
    else:
        return tensor.to(device)


def set_global_seed(config: DictConfig) -> None:
    """Set global random seed for reproducible results across all libraries.
    
    Configures random number generators for Python's random module, NumPy,
    PyTorch (CPU and CUDA), and cuDNN to ensure reproducible results.
    
    Args:
        config: Hydra configuration object. Expected to contain a 'seed' key.
                If not present, defaults to 42.
    
    Note:
        Setting deterministic=True and benchmark=False may impact performance
        but ensures reproducible results. This is particularly important for
        research and debugging purposes.
    
    Example:
        >>> config = load_config()
        >>> set_global_seed(config)
        # All subsequent random operations will be reproducible
    """
    seed = config.get("seed", 42)
    logger.info(f"Setting global seed to {seed} for reproducibility")

    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed)
    
    # Set seed for PyTorch CUDA operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        logger.debug("Set CUDA seeds for reproducibility")
    
    # Configure cuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.debug(f"Configured all random number generators with seed {seed}")

def setup_logging(
    logging_config_path: str = "conf/logging.yaml",
    default_level: int = logging.INFO,
    log_dir: Optional[str] = None
) -> None:
    """Set up logging configuration from YAML file with robust fallback handling.
    
    Configures the logging system using a YAML configuration file. If the file
    is missing, malformed, or inaccessible, falls back to a basic logging setup
    to ensure the application can continue running.
    
    Args:
        logging_config_path: Path to YAML file containing logging configuration.
                           Defaults to "conf/logging.yaml".
        default_level: Logging level for fallback configuration.
                      Defaults to logging.INFO.
        log_dir: Optional directory to redirect log files. If provided,
                all file handlers will write to this directory instead of
                their configured paths. Directory will be created if needed.
    
    Raises:
        No exceptions are raised - all errors are handled with fallback logging.
    
    Example:
        >>> setup_logging("conf/logging.yaml", logging.DEBUG, "./logs")
        # Configures logging from YAML and redirects files to ./logs/
    """
    try:
        # Attempt to read the YAML config file
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file)

        # If a custom log directory is provided, update handler file paths
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
            handlers_updated = 0
            for handler_name, handler_config in log_config.get("handlers", {}).items():
                if "filename" in handler_config:
                    # Extract filename and recompose it with the new log directory
                    filename = os.path.basename(handler_config["filename"])
                    handler_config["filename"] = os.path.join(log_dir, filename)
                    handlers_updated += 1
            
            if handlers_updated > 0:
                print(f"Redirected {handlers_updated} log handlers to directory: {log_dir}")

        # Apply logging configuration from the YAML dict
        logging.config.dictConfig(log_config)
        print(f"Successfully configured logging from {logging_config_path}")

    except (FileNotFoundError, PermissionError) as file_err:
        # If file is not found or unreadable, use fallback logging
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logging.getLogger().warning(
            f"Logging config file not found or inaccessible: {logging_config_path}. Using basic config."
        )
        logging.getLogger().debug(f"File error details: {file_err}")

    except (yaml.YAMLError, ValueError, TypeError) as parse_err:
        # If the YAML is malformed or incompatible, use fallback logging
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logging.getLogger().warning(
            f"Error parsing logging config from {logging_config_path}. Using basic config."
        )
        logging.getLogger().debug(f"Parse error details: {parse_err}")


class DocumentChunker:
    """Intelligent document chunking system for optimizing embedding quality across file types.
    
    This class provides sophisticated document chunking capabilities that address cross-modal
    retrieval ranking issues by normalizing content sizes and enhancing document context.
    Different file types receive optimized chunking strategies based on their characteristics.
    
    The chunker supports:
        - File-type specific chunk sizes (PDF, DOCX, HTML, CSV, Audio)
        - Content enhancement with contextual prefixes
        - Intelligent splitting at natural boundaries
        - Metadata preservation and enhancement
        - CSV content filtering and length limiting
    
    Attributes:
        config: Hydra configuration object
        chunk_config: File-type specific chunk size configuration
        content_config: Content enhancement configuration
        default_chunk_sizes: Fallback chunk sizes for each file type
    """
    
    def __init__(self, config_path: str = "conf") -> None:
        """Initialize chunker with configuration-based chunk sizes and enhancement settings.
        
        Args:
            config_path: Path to configuration directory containing document processing
                        settings. Defaults to "conf".
        
        Example:
            >>> chunker = DocumentChunker("conf")
            >>> documents = chunker.process_documents(docs, "pdf")
        """
        self.config = load_config(config_path)
        self.chunk_config = self.config.get('document_processing', {}).get('chunk_sizes', {})
        self.content_config = self.config.get('document_processing', {}).get('content_enhancement', {})
        
        # Default chunk sizes if not specified in config
        self.default_chunk_sizes = {
            'pdf': 1000,
            'docx': 800,
            'html': 1200,
            'csv': 400,
            'audio': 600
        }
        
        logger.info(f"Initialized DocumentChunker with chunk sizes: {self.chunk_config or self.default_chunk_sizes}")
        logger.debug(f"Content enhancement config: {self.content_config}")
    
    def get_chunk_size(self, file_type: str) -> int:
        """Get optimal chunk size for the specified file type.
        
        Retrieves the configured chunk size for a file type, falling back to
        default values if not configured.
        
        Args:
            file_type: Type of file (e.g., 'pdf', 'docx', 'html', 'csv', 'audio').
        
        Returns:
            int: Optimal chunk size in characters for the file type.
                Returns 800 as ultimate fallback if file type is unknown.
        
        Example:
            >>> chunker = DocumentChunker()
            >>> pdf_size = chunker.get_chunk_size('pdf')  # Returns 1000
            >>> unknown_size = chunker.get_chunk_size('unknown')  # Returns 800
        """
        chunk_size = self.chunk_config.get(file_type, self.default_chunk_sizes.get(file_type, 800))
        logger.debug(f"Retrieved chunk size for {file_type}: {chunk_size}")
        return chunk_size
    
    def enhance_pdf_content(self, document: Document) -> Document:
        """Enhance PDF document content with contextual information for better embeddings.
        
        Adds document and section context to PDF content to improve semantic understanding
        and retrieval quality. This helps embeddings capture document structure and hierarchy.
        
        Args:
            document: LangChain Document object containing PDF content and metadata.
        
        Returns:
            Document: Enhanced document with contextual prefixes added to content.
                     Original metadata is preserved.
        
        Example:
            >>> pdf_doc = Document(page_content="Content here", metadata={"source": "manual.pdf"})
            >>> enhanced = chunker.enhance_pdf_content(pdf_doc)
            >>> print(enhanced.page_content)  # "Document: manual.pdf\nContent here"
        """
        content = document.page_content
        metadata = document.metadata
        
        # Start with original content
        enhanced_content = content
        
        # Add heading context if available
        if metadata.get('current_heading'):
            enhanced_content = f"Section: {metadata['current_heading']}\n\n{content}"
            logger.debug(f"Added section context: {metadata['current_heading']}")
        
        # Add document type context
        doc_name = metadata.get('document_name', metadata.get('source', ''))
        if doc_name:
            enhanced_content = f"Document: {doc_name}\n{enhanced_content}"
            logger.debug(f"Added document context: {doc_name}")
        
        # Create enhanced document
        enhanced_doc = Document(
            page_content=enhanced_content,
            metadata=metadata
        )
        
        return enhanced_doc
    
    def limit_csv_content(self, document: Document) -> Optional[Document]:
        """Filter and limit CSV row content to improve embedding relevance.
        
        Processes CSV row content by filtering out very short or non-meaningful entries,
        truncating overly long content, and adding contextual prefixes for better retrieval.
        
        Args:
            document: LangChain Document containing CSV row content and metadata.
        
        Returns:
            Document or None: Enhanced document with limited content and context,
                            or None if content should be filtered out.
        
        Example:
            >>> csv_doc = Document(page_content="Short", metadata={"row_number": 1})
            >>> result = chunker.limit_csv_content(csv_doc)  # Returns None (too short)
            
            >>> csv_doc = Document(page_content="Meaningful content here", metadata={"row_number": 1})
            >>> result = chunker.limit_csv_content(csv_doc)  # Returns enhanced document
        """
        content = document.page_content
        metadata = document.metadata
        
        max_length = self.content_config.get('csv', {}).get('max_content_length', 300)
        min_meaningful_words = 3  # Require at least 3 meaningful words
        
        # Filter out very short or non-meaningful content
        words = re.findall(r'\b\w+\b', content.lower())
        if len(words) < min_meaningful_words:
            logger.debug(f"Filtered out CSV content with only {len(words)} meaningful words")
            return None
        
        # Truncate if too long
        if len(content) > max_length:
            # Try to truncate at word boundary
            truncated = content[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # If we can truncate at word boundary without losing too much
                content = truncated[:last_space] + "..."
            else:
                content = truncated + "..."
            logger.debug(f"Truncated CSV content from {len(document.page_content)} to {len(content)} characters")
        
        # Add context prefix for CSV rows
        row_num = metadata.get('row_number', '')
        source = metadata.get('source', '')
        if row_num and source:
            content = f"Data row {row_num} from {source}:\n{content}"
            logger.debug(f"Added CSV row context: row {row_num} from {source}")
        
        return Document(
            page_content=content,
            metadata=metadata
        )
    
    def chunk_large_document(self, document: Document, file_type: str) -> List[Document]:
        """Split large documents into optimally-sized chunks with intelligent boundaries.
        
        Uses RecursiveCharacterTextSplitter to break large documents into smaller chunks
        while trying to maintain semantic coherence by splitting at natural boundaries.
        Preserves and enhances metadata for each chunk.
        
        Args:
            document: LangChain Document to be chunked.
            file_type: Type of document (affects chunk size selection).
        
        Returns:
            List[Document]: List of chunked documents. If original document is small
                          enough, returns single-item list with original document.
        
        Example:
            >>> large_doc = Document(page_content="...very long content...", metadata={})
            >>> chunks = chunker.chunk_large_document(large_doc, "pdf")
            >>> print(len(chunks))  # Multiple chunks
            >>> print(chunks[0].metadata['chunk_number'])  # 1
        """
        chunk_size = self.get_chunk_size(file_type)
        content = document.page_content
        
        # If document is already small enough, return as-is
        if len(content) <= chunk_size:
            logger.debug(f"Document size ({len(content)}) within chunk limit ({chunk_size}), no chunking needed")
            return [document]
        
        logger.info(f"Chunking large {file_type} document: {len(content)} chars -> target {chunk_size} chars")
        
        # Use RecursiveCharacterTextSplitter for intelligent chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,  # Small overlap to maintain context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on natural boundaries
        )
        
        # Split the content
        chunks = text_splitter.split_text(content)
        logger.debug(f"Text splitter created {len(chunks)} chunks")
        
        # Create documents for each chunk
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            # Create enhanced metadata for each chunk
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                'chunk_number': i + 1,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk),
                'is_chunked': True
            })
            
            # Update citation source to include chunk info
            original_citation = chunk_metadata.get('citation_source', '')
            if len(chunks) > 1:
                chunk_metadata['citation_source'] = f"{original_citation} (Part {i + 1}/{len(chunks)})"
            
            chunked_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            chunked_docs.append(chunked_doc)
        
        logger.info(f"Successfully split {file_type} document into {len(chunks)} chunks (target size: {chunk_size})")
        return chunked_docs
    
    def process_documents(self, documents: List[Document], file_type: str) -> List[Document]:
        """Process documents with file-type specific optimizations and chunking.
        
        Applies comprehensive document processing including file-type specific
        enhancements, content filtering, and intelligent chunking to optimize
        embedding quality and retrieval performance.
        
        Args:
            documents: List of LangChain Documents to process.
            file_type: Type of documents being processed (pdf, csv, html, etc.).
                      Determines which optimizations are applied.
        
        Returns:
            List[Document]: List of processed and potentially chunked documents.
                          Some documents may be filtered out during processing.
        
        Example:
            >>> docs = [Document(page_content="content", metadata={})]
            >>> processed = chunker.process_documents(docs, "pdf")
            >>> print(len(processed))  # May be more than original due to chunking
        """
        processed_docs = []
        logger.info(f"Processing {len(documents)} {file_type} documents with type-specific optimizations")
        
        documents_processed = 0
        documents_filtered = 0
        
        for doc in documents:
            try:
                # Apply file-type specific enhancements
                if file_type == 'pdf':
                    doc = self.enhance_pdf_content(doc)
                    logger.debug("Applied PDF content enhancement")
                elif file_type == 'csv':
                    doc = self.limit_csv_content(doc)
                    if doc is None:  # Filtered out
                        documents_filtered += 1
                        logger.debug("Filtered out CSV document due to content quality")
                        continue
                    logger.debug("Applied CSV content limiting and enhancement")
                
                # Apply chunking if document is too large
                chunks = self.chunk_large_document(doc, file_type)
                processed_docs.extend(chunks)
                documents_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_type} document: {e}")
                logger.debug(f"Error details: {e}", exc_info=True)
                # Include original document if processing fails
                processed_docs.append(doc)
                documents_processed += 1
        
        logger.info(
            f"Completed processing: {documents_processed} documents processed, "
            f"{documents_filtered} filtered out, {len(processed_docs)} final chunks created"
        )
        return processed_docs


def preprocess_query(query: str, config_path: str = "conf") -> str:
    """Preprocess user queries to improve retrieval quality and relevance.
    
    Enhances user queries by expanding acronyms and adding contextual terms
    that improve semantic matching with document embeddings. This preprocessing
    helps bridge the gap between user language and document content.
    
    Args:
        query: Raw user query string to be processed.
        config_path: Path to configuration directory containing query preprocessing
                    settings. Defaults to "conf".
    
    Returns:
        str: Enhanced query string with expanded acronyms and contextual terms.
    
    Example:
        >>> query = "What is the program curriculum?"
        >>> enhanced = preprocess_query(query)
        >>> print(enhanced)  # "What is the program curriculum? document course training"
    """
    logger.debug(f"Preprocessing query: '{query}'")
    
    config = load_config(config_path)
    query_config = config.get('document_processing', {}).get('query_preprocessing', {})
    
    processed_query = query.strip()
    
    # Expand common acronyms if enabled
    if query_config.get('expand_acronyms', True):
        acronym_expansions = {
            # Add domain-specific acronyms as needed
            'ML': 'machine learning',
            'AI': 'artificial intelligence',
            'NLP': 'natural language processing',
            'CV': 'computer vision'
        }
        
        expansions_made = 0
        for acronym, expansion in acronym_expansions.items():
            if acronym in processed_query:
                processed_query = processed_query.replace(acronym, f"{acronym} {expansion}")
                expansions_made += 1
        
        if expansions_made > 0:
            logger.debug(f"Expanded {expansions_made} acronyms in query")
    
    # Add document-type boosting terms if enabled
    if query_config.get('boost_document_terms', True):
        document_indicators = ['programme', 'course', 'curriculum', 'syllabus', 'requirement', 'overview']
        if any(indicator in processed_query.lower() for indicator in document_indicators):
            processed_query += " document program course"
            logger.debug("Added document-type boosting terms to query")
    
    logger.debug(f"Preprocessed query result: '{processed_query}'")
    return processed_query
