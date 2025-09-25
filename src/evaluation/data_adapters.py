"""
Data adapters for converting between RAG pipeline outputs and RAGAS evaluation formats.
Provides utilities to transform data between different evaluation frameworks.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RAGPipelineToRAGAS:
    """
    Adapter to convert RAG pipeline outputs to RAGAS evaluation format.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Optional configuration for data processing
        """
        self.config = config or {}
        self.data_config = self.config.get('ragas', {}).get('data', {})
        
        # Field mappings
        self.question_field = self.data_config.get('question_field', 'question')
        self.answer_field = self.data_config.get('answer_field', 'answer')
        self.contexts_field = self.data_config.get('contexts_field', 'contexts')
        self.ground_truth_field = self.data_config.get('ground_truth_field', 'ground_truth')
        
        # Processing limits
        self.max_context_length = self.data_config.get('max_context_length', 2000)
        self.max_answer_length = self.data_config.get('max_answer_length', 1000)
        self.context_separator = self.data_config.get('context_separator', '\n\n')
        
        logger.info("RAG Pipeline to RAGAS adapter initialized")
    
    def convert_single_response(self, 
                               rag_response: Dict[str, Any],
                               ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert a single RAG pipeline response to RAGAS format.
        
        Args:
            rag_response: Response from RAG pipeline query
            ground_truth: Optional reference answer
            
        Returns:
            Dictionary in RAGAS format
        """
        try:
            # Extract question
            question = rag_response.get('query', rag_response.get('question', ''))
            if not question:
                raise ValueError("No question found in RAG response")
            
            # Extract answer
            answer = rag_response.get('answer', rag_response.get('response', ''))
            if not answer:
                raise ValueError("No answer found in RAG response")
            
            # Process answer length
            if len(answer) > self.max_answer_length:
                answer = answer[:self.max_answer_length] + '...'
                logger.warning(f"Answer truncated to {self.max_answer_length} characters")
            
            # Extract and process contexts
            contexts = self._extract_contexts(rag_response)
            
            # Build RAGAS format
            ragas_data = {
                'question': question,
                'answer': answer,
                'contexts': contexts
            }
            
            # Add ground truth if provided
            if ground_truth:
                ragas_data['ground_truth'] = ground_truth
            
            return ragas_data
            
        except Exception as e:
            logger.error(f"Failed to convert RAG response to RAGAS format: {str(e)}")
            raise
    
    def convert_batch_responses(self,
                               rag_responses: List[Dict[str, Any]],
                               ground_truths: Optional[List[str]] = None) -> Dict[str, List[Any]]:
        """
        Convert multiple RAG pipeline responses to RAGAS batch format.
        
        Args:
            rag_responses: List of RAG pipeline responses
            ground_truths: Optional list of reference answers
            
        Returns:
            Dictionary with lists for batch RAGAS evaluation
        """
        try:
            questions = []
            answers = []
            contexts_list = []
            
            for i, rag_response in enumerate(rag_responses):
                # Get ground truth for this response if available
                ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
                
                # Convert single response
                ragas_data = self.convert_single_response(rag_response, ground_truth)
                
                questions.append(ragas_data['question'])
                answers.append(ragas_data['answer'])
                contexts_list.append(ragas_data['contexts'])
            
            # Build batch format
            batch_data = {
                'question': questions,
                'answer': answers,
                'contexts': contexts_list
            }
            
            # Add ground truths if provided
            if ground_truths:
                # Ensure ground truths list matches the length
                if len(ground_truths) != len(questions):
                    logger.warning(f"Ground truths length ({len(ground_truths)}) doesn't match questions ({len(questions)})")
                    # Pad or truncate as needed
                    if len(ground_truths) < len(questions):
                        ground_truths.extend([None] * (len(questions) - len(ground_truths)))
                    else:
                        ground_truths = ground_truths[:len(questions)]
                
                batch_data['ground_truth'] = ground_truths
            
            logger.info(f"Converted {len(questions)} RAG responses to RAGAS batch format")
            return batch_data
            
        except Exception as e:
            logger.error(f"Failed to convert batch RAG responses: {str(e)}")
            raise
    
    def _extract_contexts(self, rag_response: Dict[str, Any]) -> List[str]:
        """
        Extract and process contexts from RAG pipeline response.
        
        Args:
            rag_response: RAG pipeline response
            
        Returns:
            List of processed context strings
        """
        contexts = []
        
        # Try different possible context fields
        raw_contexts = (
            rag_response.get('sources', []) or
            rag_response.get('contexts', []) or
            rag_response.get('retrieved_documents', []) or
            rag_response.get('documents', [])
        )
        
        if not raw_contexts:
            logger.warning("No contexts found in RAG response")
            return []
        
        for ctx in raw_contexts:
            if isinstance(ctx, dict):
                # Extract text content from document-like objects
                content = (
                    ctx.get('content', '') or 
                    ctx.get('page_content', '') or
                    ctx.get('text', '') or
                    str(ctx)
                )
            elif isinstance(ctx, str):
                content = ctx
            else:
                content = str(ctx)
            
            # Clean and process content
            content = content.strip()
            if content:
                # Truncate if too long
                if len(content) > self.max_context_length:
                    content = content[:self.max_context_length] + '...'
                contexts.append(content)
        
        if not contexts:
            logger.warning("No valid contexts extracted from RAG response")
        
        return contexts


class EvaluationDataLoader:
    """
    Loader for evaluation data from various formats.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the data loader."""
        self.config = config or {}
        logger.info("Evaluation data loader initialized")
    
    def load_evaluation_data(self, 
                            questions_file: str,
                            answers_file: Optional[str] = None) -> Tuple[List[str], Optional[List[str]]]:
        """
        Load questions and optional reference answers for evaluation.
        
        Args:
            questions_file: Path to questions file (JSON format)
            answers_file: Optional path to reference answers file
            
        Returns:
            Tuple of (questions, reference_answers)
        """
        try:
            # Load questions
            questions = self._load_questions(questions_file)
            
            # Load reference answers if provided
            reference_answers = None
            if answers_file:
                reference_answers = self._load_answers(answers_file, len(questions))
            
            logger.info(f"Loaded {len(questions)} questions" + 
                       (f" and {len(reference_answers)} reference answers" if reference_answers else ""))
            
            return questions, reference_answers
            
        except Exception as e:
            logger.error(f"Failed to load evaluation data: {str(e)}")
            raise
    
    def _load_questions(self, questions_file: str) -> List[str]:
        """Load questions from file."""
        questions_path = Path(questions_file)
        
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            questions = data
        elif isinstance(data, dict):
            if 'questions' in data:
                questions = data['questions']
            elif 'data' in data:
                # Handle format like {"data": [{"question": "...", ...}, ...]}
                questions = [item['question'] for item in data['data'] if 'question' in item]
            else:
                raise ValueError(f"Unsupported questions file format: {questions_file}")
        else:
            raise ValueError(f"Invalid questions file format: {questions_file}")
        
        if not questions:
            raise ValueError(f"No questions found in file: {questions_file}")
        
        return questions
    
    def _load_answers(self, answers_file: str, expected_count: int) -> List[str]:
        """Load reference answers from file."""
        answers_path = Path(answers_file)
        
        if not answers_path.exists():
            raise FileNotFoundError(f"Answers file not found: {answers_file}")
        
        with open(answers_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        answers = []
        if isinstance(data, list):
            answers = data
        elif isinstance(data, dict):
            if 'answers' in data:
                answers = data['answers']
            elif 'ground_truth' in data:
                answers = data['ground_truth']
            elif 'data' in data:
                # Handle format like {"data": [{"answer": "...", ...}, ...]}
                answers = [item.get('answer', item.get('ground_truth', '')) 
                          for item in data['data']]
            else:
                raise ValueError(f"Unsupported answers file format: {answers_file}")
        
        # Validate answer count
        if len(answers) != expected_count:
            logger.warning(f"Answer count ({len(answers)}) doesn't match question count ({expected_count})")
            
            # Pad or truncate as needed
            if len(answers) < expected_count:
                answers.extend([''] * (expected_count - len(answers)))
            else:
                answers = answers[:expected_count]
        
        return answers


class RAGASResultsProcessor:
    """
    Processor for RAGAS evaluation results.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the results processor."""
        self.config = config or {}
        logger.info("RAGAS results processor initialized")
    
    def merge_with_traditional_metrics(self,
                                     ragas_results: Dict[str, Any],
                                     traditional_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge RAGAS results with traditional evaluation metrics.
        
        Args:
            ragas_results: Results from RAGAS evaluation
            traditional_results: Results from BLEU/ROUGE evaluation
            
        Returns:
            Combined results dictionary
        """
        try:
            combined = {
                'evaluation_type': 'comprehensive',
                'timestamp': traditional_results.get('timestamp'),
                'ragas_metrics': ragas_results.get('ragas_metrics', {}),
                'traditional_metrics': {
                    'bleu_rouge': traditional_results.get('bleu_rouge_results', {}),
                    'semantic_similarity': traditional_results.get('semantic_similarity_results', {}),
                    'chromadb_metrics': traditional_results.get('chromadb_results', {})
                },
                'combined_analysis': self._create_combined_analysis(ragas_results, traditional_results)
            }
            
            logger.info("Successfully merged RAGAS and traditional evaluation results")
            return combined
            
        except Exception as e:
            logger.error(f"Failed to merge evaluation results: {str(e)}")
            raise
    
    def _create_combined_analysis(self,
                                 ragas_results: Dict[str, Any],
                                 traditional_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create combined analysis of all evaluation metrics."""
        
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Analyze RAGAS metrics
        ragas_metrics = ragas_results.get('ragas_metrics', {})
        
        if ragas_metrics.get('faithfulness', 0) >= 0.7:
            analysis['strengths'].append("High faithfulness - responses are well-grounded in retrieved context")
        elif ragas_metrics.get('faithfulness', 0) < 0.5:
            analysis['weaknesses'].append("Low faithfulness - responses may contain hallucinated information")
            analysis['recommendations'].append("Improve context relevance or adjust generation prompts")
        
        if ragas_metrics.get('answer_relevancy', 0) >= 0.7:
            analysis['strengths'].append("High answer relevancy - responses directly address questions")
        elif ragas_metrics.get('answer_relevancy', 0) < 0.5:
            analysis['weaknesses'].append("Low answer relevancy - responses may be off-topic")
            analysis['recommendations'].append("Review question understanding and response generation")
        
        # Analyze traditional metrics
        traditional = traditional_results.get('bleu_rouge_results', {})
        if traditional.get('rouge_l_f1', 0) >= 0.4:
            analysis['strengths'].append("Good ROUGE-L scores - responses align well with reference answers")
        elif traditional.get('rouge_l_f1', 0) < 0.2:
            analysis['weaknesses'].append("Low ROUGE-L scores - responses differ significantly from references")
            analysis['recommendations'].append("Review training data or generation parameters")
        
        return analysis