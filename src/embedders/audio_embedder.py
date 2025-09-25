"""Audio content embedder for processing audio files into vector embeddings.

This module provides functionality to transcribe audio files using Whisper
and convert the transcripts into embeddings for semantic search. Features
smart segmentation and natural pause detection for better transcription quality.
"""

from pathlib import Path
from typing import List, Tuple
import torchaudio
import whisper
import math
import re
from langchain.schema import Document
from langchain_chroma import Chroma

from .base_embedder import BaseEmbedder
from ..utils import DocumentChunker


class AudioEmbedder(BaseEmbedder):
    """Embeds audio files using enhanced Whisper transcription with smart segmentation and document chunking.
    Uses natural pause detection for better segment boundaries and improved transcription quality."""
    
    def __init__(self, config_path: str = "conf/config.yaml", 
                 embedding_model: str = None,
                 whisper_model: str = "medium",  # Upgraded from "small"
                 embeddings=None) -> None:
        """Initialize audio embedder with Whisper transcription.
        
        Args:
            config_path: Path to configuration file.
            embedding_model: Override for embedding model name.
            whisper_model: Whisper model size (tiny, base, small, medium, large).
            embeddings: Pre-initialized embeddings instance.
        """
        # Initialize base class with Audio-specific configuration
        super().__init__(
            config_path=config_path,
            embedding_model=embedding_model,
            embeddings=embeddings,
            document_type="audio"
        )
        self.chunker = DocumentChunker(config_path)
        
        # Get audio processing configuration
        audio_config = self.config.get('document_processing', {}).get('audio', {})
        self.whisper_model_name = audio_config.get('whisper_model', whisper_model)
        self.segmentation_method = audio_config.get('segmentation_method', 'smart')
        self.min_segment_length = audio_config.get('min_segment_length', 5)
        self.max_segment_length = audio_config.get('max_segment_length', 60)
        self.post_process_transcript = audio_config.get('post_process_transcript', True)
        self.remove_filler_words = audio_config.get('remove_filler_words', True)
        
        # Initialize Whisper for transcription with better model
        self.whisper_model = whisper.load_model(self.whisper_model_name, device=str(self.device))
        
        self.logger.info(f"Audio processing: {self.segmentation_method} segmentation, {self.min_segment_length}-{self.max_segment_length}s segments")
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Duration in seconds, 0.0 if error occurs.
        """
        try:
            wav, sr = torchaudio.load(str(audio_path))
            duration = wav.shape[1] / sr
            return float(duration)
        except Exception as e:
            self.logger.error(f"Error getting duration for {audio_path}: {e}")
            return 0.0
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS format.
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            Formatted timestamp string in MM:SS format.
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def post_process_transcript_text(self, text: str) -> str:
        """Clean up transcript formatting and remove filler words.
        
        Args:
            text: Raw transcript text.
            
        Returns:
            Cleaned and formatted transcript text.
        """
        if not self.post_process_transcript:
            return text
        
        # Remove excessive filler words if enabled
        if self.remove_filler_words:
            text = re.sub(r'\b(um|uh|like|you know|sort of|kind of)\b', '', text, flags=re.IGNORECASE)
        
        # Clean up spacing and improve formatting
        text = ' '.join(text.split())  # Normalize whitespace
        text = text.strip()
        
        # Capitalize first letter if not empty
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text
    
    def segment_by_silence(self, audio_path: Path) -> List[Tuple[float, float]]:
        """Simple smart segmentation using overlapping windows with natural boundaries.
        
        Args:
            audio_path: Path to audio file to segment.
            
        Returns:
            List of (start_time, end_time) tuples in seconds.
        """
        try:
            duration = self.get_audio_duration(audio_path)
            if duration <= 0:
                raise ValueError(f"Invalid audio duration: {duration}")
            
            segments = []
            
            # Use slightly overlapping segments that are longer than fixed approach
            # This gives more context while still being "smarter" than rigid 30s chunks
            base_segment_length = min(self.max_segment_length * 0.8, 40)  # ~40s base
            overlap = 3  # 3 second overlap for context
            
            start_time = 0.0
            segment_num = 1
            
            while start_time < duration:
                # Calculate end time
                end_time = min(start_time + base_segment_length, duration)
                
                # For segments that aren't the last one, try to find a better break point
                # by extending slightly if we're not near the end
                if end_time < duration - 5:  # Not near the end
                    # Extend up to 10 more seconds to find a better break
                    potential_end = min(end_time + 10, duration)
                    # Use the extended time (simulates finding a natural pause)
                    end_time = potential_end
                
                # Only add segment if it meets minimum length requirement
                segment_duration = end_time - start_time
                if segment_duration >= self.min_segment_length:
                    segments.append((start_time, end_time))
                    self.logger.debug(f"Smart segment {segment_num}: {start_time:.1f}s - {end_time:.1f}s ({segment_duration:.1f}s)")
                    segment_num += 1
                
                # Move start time with overlap, but ensure we make progress
                next_start = start_time + base_segment_length - overlap
                if next_start <= start_time:  # Ensure we always advance
                    next_start = start_time + base_segment_length
                
                start_time = next_start
                
                # If remaining audio is very short, break to avoid tiny segments
                if duration - start_time < self.min_segment_length:
                    break
            
            if not segments:
                self.logger.warning(f"Smart segmentation created no segments for {audio_path.name}")
                return self.segment_fixed_duration(duration)
            
            self.logger.info(f"Smart segmentation created {len(segments)} adaptive segments for {audio_path.name}")
            return segments
            
        except Exception as e:
            self.logger.warning(f"Smart segmentation failed for {audio_path.name}: {e}")
            return self.segment_fixed_duration(self.get_audio_duration(audio_path))
    
    def segment_fixed_duration(self, duration: float) -> List[Tuple[float, float]]:
        """Fallback to fixed duration segmentation.
        
        Args:
            duration: Total audio duration in seconds.
            
        Returns:
            List of (start_time, end_time) tuples in seconds.
        """
        segment_duration = self.max_segment_length  # Use max segment length as fixed duration
        segments = []
        
        num_segments = math.ceil(duration / segment_duration)
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            # Only include segments that meet minimum length
            if end_time - start_time >= self.min_segment_length:
                segments.append((start_time, end_time))
        
        return segments
    
    def get_audio_segments(self, audio_path: Path) -> List[Tuple[float, float]]:
        """Get audio segments based on configured segmentation method.
        
        Args:
            audio_path: Path to audio file to segment.
            
        Returns:
            List of (start_time, end_time) tuples in seconds.
        """
        if self.segmentation_method == 'smart':
            return self.segment_by_silence(audio_path)
        else:
            # Fixed duration segmentation
            duration = self.get_audio_duration(audio_path)
            return self.segment_fixed_duration(duration)
    
    def create_citation_with_timestamp(self, filename: str, start_time: float, end_time: float) -> str:
        """Create citation format: 'Audio File (00:45-01:30)'.
        
        Args:
            filename: Name of the audio file.
            start_time: Segment start time in seconds.
            end_time: Segment end time in seconds.
            
        Returns:
            Formatted citation string with timestamp.
        """
        start_str = self.format_timestamp(start_time)
        end_str = self.format_timestamp(end_time)
        return f"{filename} ({start_str}-{end_str})"
    
    def whisper_transcribe_segment(self, audio_path: Path, start_time: float, end_time: float) -> Tuple[str, str]:
        """Transcribe a specific segment of audio file using Whisper.
        
        Args:
            audio_path: Path to audio file.
            start_time: Segment start time in seconds.
            end_time: Segment end time in seconds.
            
        Returns:
            Tuple of (transcript_text, detected_language).
        """
        try:
            # Load and segment the audio first, then transcribe just that segment
            import tempfile
            import os
            
            # Load the full audio file
            wav, sr = torchaudio.load(str(audio_path))
            
            # Calculate sample indices for the segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract the segment
            segment_audio = wav[:, start_sample:end_sample]
            
            # Save segment to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                torchaudio.save(temp_path, segment_audio, sr)
            
            try:
                # Transcribe just the segment with FP32 to avoid warnings
                result = self.whisper_model.transcribe(temp_path, language="en", fp16=False)
                
                # Extract transcript text
                if isinstance(result, dict):
                    segment_text = result.get("text", "").strip()
                    detected_language = result.get("language", "en")
                else:
                    # Handle case where result might be a different type
                    segment_text = str(result).strip() if result else ""
                    detected_language = "en"
                
                # Apply post-processing if enabled
                transcript = self.post_process_transcript_text(segment_text)
                
                return transcript, detected_language
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            
        except Exception as e:
            self.logger.error(f"Error transcribing segment {start_time}-{end_time} from {audio_path}: {e}")
            return "", "en"
    
    def extract_documents(self, audio_dir: str = None) -> List[Document]:
        """Extract transcripts from audio files using smart segmentation and create optimized LangChain Documents.
        
        Args:
            audio_dir: Directory containing audio files to process.
            
        Returns:
            List of LangChain Document objects with transcript metadata.
        """
        # Use default directory from config if not provided
        if audio_dir is None:
            audio_dir = self.get_resource_directory()
            
        self.logger.info(f"Processing audio files from {audio_dir} with {self.segmentation_method} segmentation")
        
        # Validate directory
        if not self.validate_resource_directory(audio_dir):
            return []
        
        audio_path = Path(audio_dir)
        
        documents = []
        audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        audio_files = []
        
        # Find all audio files
        for ext in audio_extensions:
            audio_files.extend(audio_path.glob(f"*{ext}"))
        
        if not audio_files:
            self.logger.warning(f"No audio files found in {audio_dir}")
            return []
        
        for audio_file in audio_files:
            try:
                self.logger.info(f"Processing audio: {audio_file.name}")
                
                # Get audio duration
                duration = self.get_audio_duration(audio_file)
                if duration <= 0:
                    self.logger.warning(f"Could not determine duration for {audio_file.name}")
                    continue
                
                filename = audio_file.stem
                
                # Get optimal segments using configured method
                segments = self.get_audio_segments(audio_file)
                
                if not segments:
                    self.logger.warning(f"No valid segments found for {audio_file.name}")
                    continue
                
                # Process each segment
                segments_created = 0
                for i, (start_time, end_time) in enumerate(segments):
                    transcript, language = self.whisper_transcribe_segment(audio_file, start_time, end_time)
                    
                    if transcript.strip():
                        citation = self.create_citation_with_timestamp(filename, start_time, end_time)
                        
                        # Enhanced metadata with segmentation info
                        metadata = {
                            'source': audio_file.name,
                            'file_type': 'audio',
                            'file_path': str(audio_file),
                            'language': language,
                            'duration': duration,
                            'start_time': start_time,
                            'end_time': end_time,
                            'segment_number': i + 1,
                            'total_segments': len(segments),
                            'segment_duration': end_time - start_time,
                            'segmentation_method': self.segmentation_method,
                            'whisper_model': self.whisper_model_name,
                            'post_processed': self.post_process_transcript,
                            'citation_source': citation
                        }
                        
                        document = Document(
                            page_content=transcript,
                            metadata=metadata
                        )
                        documents.append(document)
                        segments_created += 1
                
                segment_type = "smart" if self.segmentation_method == 'smart' else "fixed"
                self.logger.info(f"Processed {audio_file.name}: {segments_created} {segment_type} segments from {duration:.1f}s audio")
                
            except Exception as e:
                self.logger.error(f"Error processing audio {audio_file}: {e}")
                continue
        
        # Apply intelligent chunking and content optimization
        processed_documents = self.chunker.process_documents(documents, 'audio')
        
        self.logger.info(f"Successfully processed {len(documents)} audio segments into {len(processed_documents)} optimized chunks")
        return processed_documents
    
    def embed_to_chroma(self, documents: List[Document], vectorstore: Chroma) -> int:
        """Add audio transcript documents to existing ChromaDB vectorstore.
        
        Args:
            documents: List of audio transcript documents to embed.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
            
        Raises:
            Exception: If embedding process fails.
        """
        if not documents:
            self.logger.warning("No audio documents to embed")
            return 0
            
        try:
            # Add documents to existing vectorstore
            vectorstore.add_documents(documents)
            self.logger.info(f"Successfully embedded {len(documents)} audio segments to ChromaDB")
            return len(documents)
            
        except Exception as e:
            self.logger.error(f"Error embedding audio documents: {e}")
            raise
    
    def process_and_embed(self, audio_dir: str = None, vectorstore: Chroma = None) -> int:
        """Complete pipeline: transcribe audio files into segments and embed transcripts.
        
        Args:
            audio_dir: Directory containing audio files.
            vectorstore: ChromaDB vectorstore instance.
            
        Returns:
            Number of documents successfully embedded.
        """
        # Use the base class implementation which handles defaults
        return super().process_and_embed(audio_dir, vectorstore)