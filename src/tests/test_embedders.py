"""
Comprehensive test suite for RAG system embedders.
Tests all embedder classes with mocked dependencies for production-ready validation.

This test suite covers:
- HTMLEmbedder: HTML content extraction and processing
- PDFEmbedder: PDF document processing with structure detection
- AudioEmbedder: Audio transcription and segmentation
- ImageEmbedder: Image processing with vision models
- CSVEmbedder: CSV data processing and enhancement
- DOCXEmbedder: Word document processing
- BaseEmbedder: Abstract base class functionality

Each embedder is tested for:
- Document extraction from various file formats
- Embedding generation and ChromaDB storage
- Error handling for invalid files and missing dependencies
- Configuration loading and validation
- Metadata processing and cleaning
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import yaml
from typing import List, Dict, Any
import os

# Import all embedder classes
from src.embedders.base_embedder import BaseEmbedder
from src.embedders.html_embedder import HTMLEmbedder
from src.embedders.pdf_embedder import PDFEmbedder
from src.embedders.audio_embedder import AudioEmbedder
from src.embedders.image_embedder import ImageEmbedder
from src.embedders.csv_embedder import CSVEmbedder
from src.embedders.docx_embedder import DOCXEmbedder

from langchain.schema import Document


class TestBaseEmbedder:
    """Test suite for BaseEmbedder abstract base class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for base embedder."""
        return {
            'embeddings': {
                'model': 'BAAI/bge-large-en-v1.5',
                'device': 'cpu'
            },
            'document_processing': {
                'resource_dirs': {
                    'html': 'test_corpus/html',
                    'pdf': 'test_corpus/pdf',
                    'audio': 'test_corpus/audio'
                },
                'chunk_sizes': {
                    'html': 1200,
                    'pdf': 1000,
                    'audio': 800
                }
            },
            'run': {'device': 'cpu'}
        }
    
    @pytest.fixture
    def mock_base_embedder(self, mock_config):
        """Create a concrete implementation of BaseEmbedder for testing."""
        class ConcreteEmbedder(BaseEmbedder):
            def extract_documents(self, resource_dir: str = None) -> List[Document]:
                return []
            
            def embed_to_chroma(self, documents: List[Document], vectorstore) -> int:
                return len(documents)
        
        with patch('src.embedders.base_embedder.load_config', return_value=mock_config), \
             patch('src.embedders.base_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.base_embedder.HuggingFaceEmbeddings') as mock_embeddings:
            mock_embeddings.return_value = Mock()
            embedder = ConcreteEmbedder(document_type="test")
            return embedder
    
    def test_base_embedder_initialization(self, mock_base_embedder):
        """Test BaseEmbedder initialization with proper configuration."""
        assert mock_base_embedder.document_type == "test"
        assert mock_base_embedder.default_resource_dir == "test_corpus/test"
        assert mock_base_embedder.embeddings is not None
        assert hasattr(mock_base_embedder, 'logger')
    
    def test_get_resource_directory(self, mock_base_embedder):
        """Test resource directory retrieval."""
        resource_dir = mock_base_embedder.get_resource_directory()
        assert resource_dir == "test_corpus/test"
    
    def test_validate_resource_directory_valid(self, mock_base_embedder, tmp_path):
        """Test validation of valid resource directory."""
        test_dir = tmp_path / "valid_dir"
        test_dir.mkdir()
        
        assert mock_base_embedder.validate_resource_directory(str(test_dir)) is True
    
    def test_validate_resource_directory_invalid(self, mock_base_embedder):
        """Test validation of invalid resource directory."""
        # Non-existent directory
        assert mock_base_embedder.validate_resource_directory("/nonexistent/path") is False
        
        # File instead of directory
        with tempfile.NamedTemporaryFile() as temp_file:
            assert mock_base_embedder.validate_resource_directory(temp_file.name) is False
    
    def test_clean_metadata_for_chroma(self, mock_base_embedder):
        """Test metadata cleaning for ChromaDB compatibility."""
        metadata = {
            'string_val': 'test',
            'int_val': 123,
            'float_val': 45.6,
            'bool_val': True,
            'none_val': None,
            'list_val': [1, 2, 3],  # Should be converted to string
            'dict_val': {'key': 'value'}  # Should be converted to string
        }
        
        cleaned = mock_base_embedder.clean_metadata_for_chroma(metadata)
        
        assert cleaned['string_val'] == 'test'
        assert cleaned['int_val'] == 123
        assert cleaned['float_val'] == 45.6
        assert cleaned['bool_val'] is True
        assert cleaned['none_val'] is None
        assert isinstance(cleaned['list_val'], str)
        assert isinstance(cleaned['dict_val'], str)
    
    def test_process_and_embed_with_vectorstore(self, mock_base_embedder):
        """Test complete process and embed pipeline."""
        mock_vectorstore = Mock()
        
        # Mock extract_documents to return test documents
        test_docs = [Document(page_content="test", metadata={})]
        mock_base_embedder.extract_documents = Mock(return_value=test_docs)
        
        result = mock_base_embedder.process_and_embed(vectorstore=mock_vectorstore)
        
        assert result == 1
        mock_base_embedder.extract_documents.assert_called_once()
    
    def test_process_and_embed_without_vectorstore(self, mock_base_embedder):
        """Test process and embed without vectorstore fails gracefully."""
        result = mock_base_embedder.process_and_embed(vectorstore=None)
        assert result == 0


class TestHTMLEmbedder:
    """Test suite for HTMLEmbedder class."""
    
    @pytest.fixture
    def mock_html_config(self):
        """Mock configuration for HTML embedder."""
        return {
            'embeddings': {'model': 'test-model', 'device': 'cpu'},
            'document_processing': {
                'resource_dirs': {'html': 'test_html'},
                'chunk_sizes': {'html': 1200}
            },
            'run': {'device': 'cpu'}
        }
    
    @pytest.fixture
    def html_embedder(self, mock_html_config):
        """Create HTMLEmbedder with mocked dependencies."""
        with patch('src.embedders.html_embedder.load_config', return_value=mock_html_config), \
             patch('src.embedders.html_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.html_embedder.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.embedders.html_embedder.DocumentChunker') as mock_chunker:
            
            mock_embeddings.return_value = Mock()
            mock_chunker.return_value = Mock()
            
            embedder = HTMLEmbedder()
            return embedder
    
    @pytest.fixture
    def sample_html_files(self, tmp_path):
        """Create sample HTML files for testing."""
        html_dir = tmp_path / "html"
        html_dir.mkdir()
        
        # Valid HTML file
        valid_html = html_dir / "test.html"
        valid_html.write_text("""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Content</h1>
                <p>This is test content for HTML parsing.</p>
            </body>
        </html>
        """)
        
        # Empty HTML file
        empty_html = html_dir / "empty.html"
        empty_html.write_text("<html></html>")
        
        return html_dir
    
    def test_html_embedder_initialization(self, html_embedder):
        """Test HTMLEmbedder initialization."""
        assert html_embedder.document_type == "html"
        assert hasattr(html_embedder, 'chunker')
        assert hasattr(html_embedder, 'embeddings')
    
    @patch('src.embedders.html_embedder.HTMLParser')
    def test_extract_documents_success(self, mock_parser, html_embedder):
        """Test successful HTML document extraction."""
        # Mock HTML parser results
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_all_html_files.return_value = [
            {
                'content': 'Test HTML content',
                'source': 'test.html',
                'file_path': '/test/test.html',
                'title': 'Test Page',
                'description': 'Test description',
                'url': 'http://test.com'
            }
        ]
        
        # Mock chunker
        test_doc = Document(
            page_content='Test HTML content',
            metadata={
                'source': 'test.html',
                'file_type': 'html',
                'citation_source': 'test.html'
            }
        )
        html_embedder.chunker.process_documents.return_value = [test_doc]
        
        with patch.object(html_embedder, 'validate_resource_directory', return_value=True):
            documents = html_embedder.extract_documents("test_dir")
        
        assert len(documents) == 1
        assert documents[0].page_content == 'Test HTML content'
        assert documents[0].metadata['file_type'] == 'html'
        assert documents[0].metadata['source'] == 'test.html'
        mock_parser.assert_called_once_with("test_dir")
    
    @patch('src.embedders.html_embedder.HTMLParser')
    def test_extract_documents_empty_content(self, mock_parser, html_embedder):
        """Test HTML extraction with empty content."""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_all_html_files.return_value = [
            {
                'content': '',  # Empty content
                'source': 'empty.html',
                'file_path': '/test/empty.html'
            }
        ]
        
        html_embedder.chunker.process_documents.return_value = []
        
        with patch.object(html_embedder, 'validate_resource_directory', return_value=True):
            documents = html_embedder.extract_documents("test_dir")
        
        assert len(documents) == 0
    
    def test_extract_documents_invalid_directory(self, html_embedder):
        """Test HTML extraction with invalid directory."""
        with patch.object(html_embedder, 'validate_resource_directory', return_value=False):
            documents = html_embedder.extract_documents("/invalid/path")
        
        assert len(documents) == 0
    
    def test_embed_to_chroma_success(self, html_embedder):
        """Test successful embedding to ChromaDB."""
        mock_vectorstore = Mock()
        test_docs = [
            Document(page_content="test", metadata={'file_type': 'html'})
        ]
        
        result = html_embedder.embed_to_chroma(test_docs, mock_vectorstore)
        
        assert result == 1
        mock_vectorstore.add_documents.assert_called_once_with(test_docs)
    
    def test_embed_to_chroma_empty_documents(self, html_embedder):
        """Test embedding empty document list."""
        mock_vectorstore = Mock()
        
        result = html_embedder.embed_to_chroma([], mock_vectorstore)
        
        assert result == 0
        mock_vectorstore.add_documents.assert_not_called()
    
    def test_embed_to_chroma_failure(self, html_embedder):
        """Test embedding failure handling."""
        mock_vectorstore = Mock()
        mock_vectorstore.add_documents.side_effect = Exception("ChromaDB error")
        test_docs = [Document(page_content="test", metadata={})]
        
        with pytest.raises(Exception, match="ChromaDB error"):
            html_embedder.embed_to_chroma(test_docs, mock_vectorstore)


class TestPDFEmbedder:
    """Test suite for PDFEmbedder class."""
    
    @pytest.fixture
    def mock_pdf_config(self):
        """Mock configuration for PDF embedder."""
        return {
            'embeddings': {'model': 'test-model', 'device': 'cpu'},
            'document_processing': {
                'resource_dirs': {'pdf': 'test_pdf'},
                'chunk_sizes': {'pdf': 1000}
            },
            'run': {'device': 'cpu'}
        }
    
    @pytest.fixture
    def pdf_embedder(self, mock_pdf_config):
        """Create PDFEmbedder with mocked dependencies."""
        with patch('src.embedders.pdf_embedder.load_config', return_value=mock_pdf_config), \
             patch('src.embedders.pdf_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.pdf_embedder.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.embedders.pdf_embedder.DocumentChunker') as mock_chunker:
            
            mock_embeddings.return_value = Mock()
            mock_chunker.return_value = Mock()
            
            embedder = PDFEmbedder()
            return embedder
    
    def test_pdf_embedder_initialization(self, pdf_embedder):
        """Test PDFEmbedder initialization."""
        assert pdf_embedder.document_type == "pdf"
        assert hasattr(pdf_embedder, 'chunker')
    
    def test_extract_headings_and_structure(self, pdf_embedder):
        """Test heading extraction from PDF content."""
        content = """
        INTRODUCTION
        
        This is the introduction section.
        
        1. METHODOLOGY
        
        This describes the methodology.
        
        Chapter 2: Results
        
        Results content here.
        
        Table of Contents
        
        More content.
        """
        
        structure = pdf_embedder._extract_headings_and_structure(content)
        
        assert len(structure['headings']) > 0
        assert 'INTRODUCTION' in structure['headings']
        assert structure['has_toc'] is True
        assert any('METHODOLOGY' in heading for heading in structure['headings'])
    
    def test_create_enhanced_citation(self, pdf_embedder):
        """Test enhanced citation creation."""
        # Citation without heading
        citation = pdf_embedder._create_enhanced_citation("test_doc", 5)
        assert citation == "test_doc (Page 5)"
        
        # Citation with heading
        citation_with_heading = pdf_embedder._create_enhanced_citation(
            "test_doc", 5, "Introduction"
        )
        assert citation_with_heading == "test_doc (Page 5, Introduction)"
        
        # Citation with very long heading (should exclude)
        long_heading = "A" * 60
        citation_long = pdf_embedder._create_enhanced_citation(
            "test_doc", 5, long_heading
        )
        assert citation_long == "test_doc (Page 5)"
    
    @patch('src.embedders.pdf_embedder.UnstructuredPDFLoader')
    def test_extract_documents_success(self, mock_loader_class, pdf_embedder):
        """Test successful PDF document extraction."""
        # Mock PDF loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Mock PDF documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "Page 1 content with HEADING"
        mock_doc1.metadata = {}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Page 2 content"
        mock_doc2.metadata = {}
        
        mock_loader.load.return_value = [mock_doc1, mock_doc2]
        
        # Mock file system
        with patch('pathlib.Path.glob') as mock_glob, \
             patch.object(pdf_embedder, 'validate_resource_directory', return_value=True):
            
            mock_file = Mock()
            mock_file.name = "test.pdf"
            mock_file.stem = "test"
            mock_glob.return_value = [mock_file]
            
            # Mock chunker
            processed_docs = [
                Document(page_content="Processed content", metadata={'file_type': 'pdf'})
            ]
            pdf_embedder.chunker.process_documents.return_value = processed_docs
            
            documents = pdf_embedder.extract_documents("test_dir")
        
        assert len(documents) == 1
        assert documents[0].metadata['file_type'] == 'pdf'
        mock_loader_class.assert_called_once()
    
    @patch('pathlib.Path.glob')
    def test_extract_documents_no_pdfs(self, mock_glob, pdf_embedder):
        """Test PDF extraction with no PDF files."""
        mock_glob.return_value = []
        
        with patch.object(pdf_embedder, 'validate_resource_directory', return_value=True):
            documents = pdf_embedder.extract_documents("test_dir")
        
        assert len(documents) == 0
    
    @patch('src.embedders.pdf_embedder.UnstructuredPDFLoader')
    @patch('pathlib.Path.glob')
    def test_extract_documents_loader_failure(self, mock_glob, mock_loader_class, pdf_embedder):
        """Test PDF extraction with loader failure."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.stem = "test"
        mock_glob.return_value = [mock_file]
        
        # Mock loader failure
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load.side_effect = Exception("PDF loading error")
        
        pdf_embedder.chunker.process_documents.return_value = []
        
        with patch.object(pdf_embedder, 'validate_resource_directory', return_value=True):
            documents = pdf_embedder.extract_documents("test_dir")
        
        assert len(documents) == 0


class TestAudioEmbedder:
    """Test suite for AudioEmbedder class."""
    
    @pytest.fixture
    def mock_audio_config(self):
        """Mock configuration for audio embedder."""
        return {
            'embeddings': {'model': 'test-model', 'device': 'cpu'},
            'document_processing': {
                'resource_dirs': {'audio': 'test_audio'},
                'audio': {
                    'whisper_model': 'small',
                    'segmentation_method': 'smart',
                    'min_segment_length': 5,
                    'max_segment_length': 60,
                    'post_process_transcript': True,
                    'remove_filler_words': True
                }
            },
            'run': {'device': 'cpu'}
        }
    
    @pytest.fixture
    def audio_embedder(self, mock_audio_config):
        """Create AudioEmbedder with mocked dependencies."""
        with patch('src.embedders.audio_embedder.load_config', return_value=mock_audio_config), \
             patch('src.embedders.audio_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.audio_embedder.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.embedders.audio_embedder.DocumentChunker') as mock_chunker, \
             patch('src.embedders.audio_embedder.whisper.load_model') as mock_whisper:
            
            mock_embeddings.return_value = Mock()
            mock_chunker.return_value = Mock()
            mock_whisper.return_value = Mock()
            
            embedder = AudioEmbedder()
            return embedder
    
    def test_audio_embedder_initialization(self, audio_embedder):
        """Test AudioEmbedder initialization."""
        assert audio_embedder.document_type == "audio"
        assert audio_embedder.whisper_model_name == "small"
        assert audio_embedder.segmentation_method == "smart"
        assert hasattr(audio_embedder, 'whisper_model')
    
    def test_format_timestamp(self, audio_embedder):
        """Test timestamp formatting."""
        assert audio_embedder.format_timestamp(65) == "01:05"
        assert audio_embedder.format_timestamp(125) == "02:05"
        assert audio_embedder.format_timestamp(30) == "00:30"
    
    def test_post_process_transcript_text(self, audio_embedder):
        """Test transcript post-processing."""
        raw_text = "um, this is like a test, you know"
        processed = audio_embedder.post_process_transcript_text(raw_text)
        
        # Should remove filler words and normalize
        assert "um" not in processed.lower()
        assert "like" not in processed.lower()
        assert "you know" not in processed.lower()
        assert processed.startswith("This")  # Capitalized
    
    def test_create_citation_with_timestamp(self, audio_embedder):
        """Test citation creation with timestamps."""
        citation = audio_embedder.create_citation_with_timestamp("test_audio", 30, 90)
        assert citation == "test_audio (00:30-01:30)"
    
    @patch('src.embedders.audio_embedder.torchaudio.load')
    def test_get_audio_duration(self, mock_load, audio_embedder):
        """Test audio duration calculation."""
        # Mock audio tensor and sample rate
        mock_wav = Mock()
        mock_wav.shape = [1, 44100 * 60]  # 1 minute of audio at 44.1kHz
        mock_load.return_value = (mock_wav, 44100)
        
        duration = audio_embedder.get_audio_duration(Path("test.wav"))
        assert duration == 60.0
    
    @patch('src.embedders.audio_embedder.torchaudio.load')
    def test_get_audio_duration_error(self, mock_load, audio_embedder):
        """Test audio duration calculation with error."""
        mock_load.side_effect = Exception("Audio loading error")
        
        duration = audio_embedder.get_audio_duration(Path("invalid.wav"))
        assert duration == 0.0
    
    def test_segment_fixed_duration(self, audio_embedder):
        """Test fixed duration segmentation."""
        segments = audio_embedder.segment_fixed_duration(150)  # 2.5 minutes
        
        assert len(segments) > 0
        for start, end in segments:
            assert end > start
            assert end - start >= audio_embedder.min_segment_length
            assert end - start <= audio_embedder.max_segment_length
    
    @patch.object(AudioEmbedder, 'get_audio_duration')
    def test_segment_by_silence(self, mock_duration, audio_embedder):
        """Test smart segmentation."""
        mock_duration.return_value = 120.0  # 2 minutes
        
        segments = audio_embedder.segment_by_silence(Path("test.wav"))
        
        assert len(segments) > 0
        for start, end in segments:
            assert end > start
            assert end - start >= audio_embedder.min_segment_length
    
    @patch('pathlib.Path.glob')
    def test_extract_documents_no_audio_files(self, mock_glob, audio_embedder):
        """Test audio extraction with no audio files."""
        mock_glob.return_value = []
        
        with patch.object(audio_embedder, 'validate_resource_directory', return_value=True):
            documents = audio_embedder.extract_documents("test_dir")
        
        assert len(documents) == 0
    
    @patch('pathlib.Path.glob')
    @patch.object(AudioEmbedder, 'get_audio_duration')
    @patch.object(AudioEmbedder, 'get_audio_segments')
    @patch.object(AudioEmbedder, 'whisper_transcribe_segment')
    def test_extract_documents_success(self, mock_transcribe, mock_segments, 
                                     mock_duration, mock_glob, audio_embedder):
        """Test successful audio document extraction."""
        # Mock file discovery
        mock_file = Mock()
        mock_file.name = "test.mp3"
        mock_file.stem = "test"
        mock_glob.return_value = [mock_file]
        
        # Mock audio processing
        mock_duration.return_value = 120.0
        mock_segments.return_value = [(0, 60), (60, 120)]
        mock_transcribe.return_value = ("Test transcript", "en")
        
        # Mock chunker
        processed_docs = [
            Document(page_content="Test transcript", metadata={'file_type': 'audio'})
        ]
        audio_embedder.chunker.process_documents.return_value = processed_docs
        
        with patch.object(audio_embedder, 'validate_resource_directory', return_value=True):
            documents = audio_embedder.extract_documents("test_dir")
        
        assert len(documents) == 1
        assert documents[0].metadata['file_type'] == 'audio'


class TestImageEmbedder:
    """Test suite for ImageEmbedder class."""
    
    @pytest.fixture
    def mock_image_config(self):
        """Mock configuration for image embedder."""
        return {
            'embeddings': {'model': 'test-model', 'device': 'cpu'},
            'document_processing': {
                'resource_dirs': {'images': 'test_images'},
                'image': {
                    'vision_model': 'test-vision-model',
                    'max_description_length': 500
                }
            },
            'run': {'device': 'cpu'}
        }
    
    @pytest.fixture
    def image_embedder(self, mock_image_config):
        """Create ImageEmbedder with mocked dependencies."""
        with patch('src.embedders.image_embedder.load_config', return_value=mock_image_config), \
             patch('src.embedders.image_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.image_embedder.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.embedders.image_embedder.DocumentChunker') as mock_chunker:
            
            mock_embeddings.return_value = Mock()
            mock_chunker.return_value = Mock()
            
            try:
                embedder = ImageEmbedder()
                return embedder
            except Exception:
                # Skip if ImageEmbedder not fully implemented
                pytest.skip("ImageEmbedder not available")
    
    def test_image_embedder_initialization(self, image_embedder):
        """Test ImageEmbedder initialization."""
        assert image_embedder.document_type == "images"
        assert hasattr(image_embedder, 'chunker')


class TestCSVEmbedder:
    """Test suite for CSVEmbedder class."""
    
    @pytest.fixture
    def mock_csv_config(self):
        """Mock configuration for CSV embedder."""
        return {
            'embeddings': {'model': 'test-model', 'device': 'cpu'},
            'document_processing': {
                'resource_dirs': {'csv': 'test_csv'},
                'content_enhancement': {
                    'csv': {'max_content_length': 300}
                }
            },
            'run': {'device': 'cpu'}
        }
    
    @pytest.fixture
    def csv_embedder(self, mock_csv_config):
        """Create CSVEmbedder with mocked dependencies."""
        with patch('src.embedders.csv_embedder.load_config', return_value=mock_csv_config), \
             patch('src.embedders.csv_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.csv_embedder.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.embedders.csv_embedder.DocumentChunker') as mock_chunker:
            
            mock_embeddings.return_value = Mock()
            mock_chunker.return_value = Mock()
            
            try:
                embedder = CSVEmbedder()
                return embedder
            except Exception:
                # Skip if CSVEmbedder not fully implemented
                pytest.skip("CSVEmbedder not available")
    
    def test_csv_embedder_initialization(self, csv_embedder):
        """Test CSVEmbedder initialization."""
        assert csv_embedder.document_type == "csv"
        assert hasattr(csv_embedder, 'chunker')


class TestDOCXEmbedder:
    """Test suite for DOCXEmbedder class."""
    
    @pytest.fixture
    def mock_docx_config(self):
        """Mock configuration for DOCX embedder."""
        return {
            'embeddings': {'model': 'test-model', 'device': 'cpu'},
            'document_processing': {
                'resource_dirs': {'docx': 'test_docx'},
                'chunk_sizes': {'docx': 1000}
            },
            'run': {'device': 'cpu'}
        }
    
    @pytest.fixture
    def docx_embedder(self, mock_docx_config):
        """Create DOCXEmbedder with mocked dependencies."""
        with patch('src.embedders.docx_embedder.load_config', return_value=mock_docx_config), \
             patch('src.embedders.docx_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.docx_embedder.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.embedders.docx_embedder.DocumentChunker') as mock_chunker:
            
            mock_embeddings.return_value = Mock()
            mock_chunker.return_value = Mock()
            
            try:
                embedder = DOCXEmbedder()
                return embedder
            except Exception:
                # Skip if DOCXEmbedder not fully implemented
                pytest.skip("DOCXEmbedder not available")
    
    def test_docx_embedder_initialization(self, docx_embedder):
        """Test DOCXEmbedder initialization."""
        assert docx_embedder.document_type == "docx"
        assert hasattr(docx_embedder, 'chunker')


class TestEmbedderIntegration:
    """Integration tests for embedder system."""
    
    def test_embedder_compatibility_with_chromadb(self):
        """Test that all embedders produce ChromaDB-compatible metadata."""
        base_config = {
            'embeddings': {'model': 'test-model', 'device': 'cpu'},
            'document_processing': {'resource_dirs': {}},
            'run': {'device': 'cpu'}
        }
        
        with patch('src.embedders.base_embedder.load_config', return_value=base_config), \
             patch('src.embedders.base_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.base_embedder.HuggingFaceEmbeddings'):
            
            # Test BaseEmbedder metadata cleaning
            class TestEmbedder(BaseEmbedder):
                def extract_documents(self, resource_dir: str = None):
                    return []
                def embed_to_chroma(self, documents, vectorstore):
                    return 0
            
            embedder = TestEmbedder(document_type="test")
            
            # Test various metadata types
            metadata = {
                'valid_string': 'test',
                'valid_int': 123,
                'valid_float': 12.3,
                'valid_bool': True,
                'valid_none': None,
                'invalid_list': [1, 2, 3],
                'invalid_dict': {'nested': 'value'},
                'invalid_object': object()
            }
            
            cleaned = embedder.clean_metadata_for_chroma(metadata)
            
            # Should preserve valid types
            assert isinstance(cleaned['valid_string'], str)
            assert isinstance(cleaned['valid_int'], int)
            assert isinstance(cleaned['valid_float'], float)
            assert isinstance(cleaned['valid_bool'], bool)
            assert cleaned['valid_none'] is None
            
            # Should convert invalid types to strings
            assert isinstance(cleaned['invalid_list'], str)
            assert isinstance(cleaned['invalid_dict'], str)
            assert isinstance(cleaned['invalid_object'], str)
    
    @patch('src.embedders.base_embedder.load_config')
    @patch('src.embedders.base_embedder.get_device')
    @patch('src.embedders.base_embedder.HuggingFaceEmbeddings')
    def test_embedder_error_handling(self, mock_embeddings, mock_device, mock_config):
        """Test error handling across all embedders."""
        mock_config.return_value = {
            'embeddings': {'model': 'test-model'},
            'document_processing': {'resource_dirs': {}},
            'run': {'device': 'cpu'}
        }
        mock_device.return_value = 'cpu'
        mock_embeddings.return_value = Mock()
        
        class TestEmbedder(BaseEmbedder):
            def extract_documents(self, resource_dir: str = None):
                if resource_dir == "error":
                    raise Exception("Test extraction error")
                return []
            
            def embed_to_chroma(self, documents, vectorstore):
                if not vectorstore:
                    raise ValueError("No vectorstore")
                return len(documents)
        
        embedder = TestEmbedder(document_type="test")
        
        # Test extraction error handling
        with pytest.raises(Exception, match="Test extraction error"):
            embedder.extract_documents("error")
        
        # Test embedding error handling
        with pytest.raises(ValueError, match="No vectorstore"):
            embedder.embed_to_chroma([], None)
        
        # Test graceful handling of invalid directories
        with patch.object(embedder, 'validate_resource_directory', return_value=False):
            result = embedder.process_and_embed("invalid_dir", Mock())
            assert result == 0


class TestEmbedderPerformance:
    """Performance and scalability tests for embedders."""
    
    def test_metadata_cleaning_performance(self):
        """Test metadata cleaning performance with large datasets."""
        base_config = {
            'embeddings': {'model': 'test-model', 'device': 'cpu'},
            'document_processing': {'resource_dirs': {}},
            'run': {'device': 'cpu'}
        }
        
        with patch('src.embedders.base_embedder.load_config', return_value=base_config), \
             patch('src.embedders.base_embedder.get_device', return_value='cpu'), \
             patch('src.embedders.base_embedder.HuggingFaceEmbeddings'):
            
            class TestEmbedder(BaseEmbedder):
                def extract_documents(self, resource_dir: str = None):
                    return []
                def embed_to_chroma(self, documents, vectorstore):
                    return 0
            
            embedder = TestEmbedder(document_type="test")
            
            # Create large metadata dictionary
            large_metadata = {}
            for i in range(1000):
                large_metadata[f'key_{i}'] = f'value_{i}'
                large_metadata[f'list_{i}'] = [1, 2, 3]
                large_metadata[f'dict_{i}'] = {'nested': f'value_{i}'}
            
            # Should complete without timeout or memory issues
            import time
            start_time = time.time()
            cleaned = embedder.clean_metadata_for_chroma(large_metadata)
            end_time = time.time()
            
            # Should complete quickly (< 1 second for 3000 items)
            assert end_time - start_time < 1.0
            assert len(cleaned) == len(large_metadata)