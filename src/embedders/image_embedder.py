"""Image content embedder for processing image files into vector embeddings.

This module provides functionality to analyze images using llama3.2-vision model
to generate descriptions, then convert those descriptions into embeddings for
semantic search of visual content.
"""

from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from .base_embedder import BaseEmbedder


class ImageEmbedder(BaseEmbedder):
    """Embeds images using llama3.2-vision for description extraction, then BAAI for embedding."""
    
    def __init__(self, config_path: str = "conf/config.yaml", 
                 embedding_model: str = None,
                 vision_model: str = "llama3.2-vision:latest",
                 ollama_base_url: str = "http://localhost:11434",
                 embeddings=None) -> None:
        """Initialize image embedder with vision model.
        
        Args:
            config_path: Path to configuration file.
            embedding_model: Override for embedding model name.
            vision_model: Ollama vision model for image analysis.
            ollama_base_url: Base URL for Ollama API.
            embeddings: Pre-initialized embeddings instance.
        """
        # Initialize base class with Images-specific configuration
        super().__init__(
            config_path=config_path,
            embedding_model=embedding_model,
            embeddings=embeddings,
            document_type="images"
        )
        
        # Initialize vision model via LangChain Ollama
        self.vision_model = ChatOllama(
            model=vision_model,
            base_url=ollama_base_url,
            temperature=0.1
        )
    
    def describe_image(self, image_path: Path) -> str:
        """Use llama3.2-vision to generate description of image.
        
        Args:
            image_path: Path to image file to analyze.
            
        Returns:
            Comprehensive description of the image content.
        """
        try:
            # Create an enhanced prompt for comprehensive image analysis
            prompt = """Analyze this image carefully and extract ALL textual information and visual details. For diagrams, flowcharts, infographics, or any image with text, describe:

1. ALL visible text content (headings, labels, descriptions, captions, numbers, percentages)
   - Read and transcribe every piece of text you can see
   - Include exact phrases, technical terms, and specific details
   - Note any acronyms, abbreviations, or specialized terminology

2. Process flow and connections between elements
   - How different parts relate to each other
   - Sequential steps or hierarchical relationships
   - Arrows, lines, or other connecting elements

3. Specific data and details mentioned
   - Numbers, statistics, timeframes, percentages
   - Names of programs, organizations, or initiatives
   - Technical specifications or requirements

4. Key concepts and terminology shown
   - Main topics or subjects covered
   - Technical or domain-specific language
   - Important keywords for searchability

5. The main purpose/message of the image
   - What is this image trying to communicate?
   - What would someone learn from viewing this?

Be comprehensive and specific - include exact text where visible. This description will be used for semantic search, so include all details that someone might search for."""

            # For LangChain Ollama with vision, we need to use the image in the message
            # Note: This assumes the vision model can handle file paths or base64
            # You may need to adjust based on your Ollama setup
            
            # Read image as base64 for vision model
            import base64
            with open(image_path, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create message with image
            from langchain.schema import HumanMessage
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = self.vision_model.invoke([message])
            description = response.content.strip()
            
            if not description:
                self.logger.warning(f"Empty description for {image_path.name}")
                return f"Image file: {image_path.name}"
            
            return description
            
        except Exception as e:
            self.logger.error(f"Error describing image {image_path}: {e}")
            # Fallback description
            return f"Image file: {image_path.name} (description unavailable)"
    
    def extract_documents(self, image_dir: str = None) -> List[Document]:
        """Extract descriptions from image files and create LangChain Documents with citation metadata.
        
        Args:
            image_dir: Directory containing image files to process.
            
        Returns:
            List of LangChain Document objects with image descriptions and metadata.
        """
        # Use default directory from config if not provided
        if image_dir is None:
            image_dir = self.get_resource_directory()
            
        self.logger.info(f"Processing image files from {image_dir}")
        
        # Validate directory
        if not self.validate_resource_directory(image_dir):
            return []
        
        image_path = Path(image_dir)
        
        documents = []
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        image_files = []
        
        # Find all image files
        for ext in image_extensions:
            image_files.extend(image_path.glob(f"*{ext}"))
            image_files.extend(image_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.warning(f"No image files found in {image_dir}")
            return []
        
        for image_file in image_files:
            try:
                self.logger.info(f"Processing image: {image_file.name}")
                
                # Generate description using vision model
                description = self.describe_image(image_file)
                
                if not description.strip():
                    self.logger.warning(f"Empty description for {image_file.name}")
                    continue
                
                # Create LangChain Document with description as content
                metadata = {
                    'source': image_file.name,
                    'file_type': 'image',
                    'file_path': str(image_file),
                    'image_format': image_file.suffix.lower(),
                    'vision_model': 'llama3.2-vision:latest',
                    # For citation: use filename without extension
                    'citation_source': image_file.stem
                }
                
                document = Document(
                    page_content=description,
                    metadata=metadata
                )
                documents.append(document)
                
                self.logger.info(f"Described {image_file.name}: {len(description)} characters")
                
            except Exception as e:
                self.logger.error(f"Error processing image {image_file}: {e}")
                continue
        
        self.logger.info(f"Successfully processed {len(documents)} image files")
        return documents
    
    def embed_to_chroma(self, documents: List[Document], vectorstore: Chroma) -> int:
        """Add image description documents to existing ChromaDB vectorstore.
        
        Args:
            documents: List of image description documents to embed.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
            
        Raises:
            Exception: If embedding process fails.
        """
        if not documents:
            self.logger.warning("No image documents to embed")
            return 0
            
        try:
            # Add documents to existing vectorstore
            vectorstore.add_documents(documents)
            self.logger.info(f"Successfully embedded {len(documents)} image descriptions to ChromaDB")
            return len(documents)
            
        except Exception as e:
            self.logger.error(f"Error embedding image documents: {e}")
            raise
    
    def process_and_embed(self, image_dir: str = None, vectorstore: Chroma = None) -> int:
        """Complete pipeline: describe images and embed descriptions.
        
        Args:
            image_dir: Directory containing image files.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
        """
        # Use the base class implementation which handles defaults
        return super().process_and_embed(image_dir, vectorstore)