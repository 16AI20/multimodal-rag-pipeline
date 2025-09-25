"""
RAGAS (RAG Assessment) evaluator for comprehensive RAG system evaluation.
Provides specialized metrics for RAG systems including faithfulness, answer relevancy,
context precision, and context recall.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optional RAGAS imports - gracefully handle if not installed
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy, 
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    logger.info("RAGAS library loaded successfully")
except ImportError as e:
    RAGAS_AVAILABLE = False
    logger.warning(f"RAGAS not available: {e}. Install with: pip install ragas>=0.1.0")
    
    # Create dummy objects for type hints
    class Dataset:
        pass


@dataclass
class RAGASResult:
    """Container for RAGAS evaluation results."""
    
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_similarity: Optional[float] = None
    answer_correctness: Optional[float] = None
    
    # Aggregate metrics
    overall_score: Optional[float] = None
    metric_count: int = 0
    
    # Metadata
    evaluation_time: Optional[float] = None
    model_info: Optional[Dict[str, str]] = None
    errors: Optional[List[str]] = None


class RAGASEvaluator:
    """
    RAGAS-based evaluator for RAG systems.
    
    Provides comprehensive evaluation using RAGAS metrics specifically designed
    for assessing Retrieval-Augmented Generation systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAGAS evaluator.
        
        Args:
            config: Configuration dictionary containing RAGAS settings
        """
        self.config = config
        self.ragas_config = config.get('ragas', {})
        
        # Check if RAGAS is available
        if not RAGAS_AVAILABLE:
            if not self.ragas_config.get('fallback_on_error', True):
                raise ImportError("RAGAS library not available. Install with: pip install ragas>=0.1.0")
            logger.warning("RAGAS not available, evaluator will return None results")
            return
            
        # Initialize metrics based on configuration
        self.enabled_metrics = self._get_enabled_metrics()
        
        # Set up models for evaluation
        self._setup_models()
        
        logger.info(f"RAGAS evaluator initialized with metrics: {list(self.enabled_metrics.keys())}")
    
    def _get_enabled_metrics(self) -> Dict[str, Any]:
        """Get enabled RAGAS metrics based on configuration."""
        if not RAGAS_AVAILABLE:
            return {}
            
        metrics_config = self.ragas_config.get('metrics', {})
        enabled = {}
        
        if metrics_config.get('faithfulness', False):
            enabled['faithfulness'] = faithfulness
            
        if metrics_config.get('answer_relevancy', False):
            enabled['answer_relevancy'] = answer_relevancy
            
        if metrics_config.get('context_precision', False):
            enabled['context_precision'] = context_precision
            
        if metrics_config.get('context_recall', False):
            enabled['context_recall'] = context_recall
            
        if metrics_config.get('answer_similarity', False):
            enabled['answer_similarity'] = answer_similarity
            
        if metrics_config.get('answer_correctness', False):
            enabled['answer_correctness'] = answer_correctness
            
        return enabled
    
    def _setup_models(self):
        """Set up models for RAGAS evaluation based on configuration."""
        if not RAGAS_AVAILABLE:
            return
            
        models_config = self.ragas_config.get('models', {})
        
        # TODO: Configure RAGAS models based on provider
        # This will depend on the specific RAGAS version and model setup
        # For now, we'll use default models
        
        self.embedding_model = models_config.get('embedding_model', 'BAAI/bge-large-en-v1.5')
        self.llm_provider = models_config.get('llm_provider', 'ollama')
        self.llm_model = models_config.get('llm_model', 'llama3.1:8b')
        
        logger.info(f"RAGAS configured with embedding: {self.embedding_model}, LLM: {self.llm_model}")
    
    def evaluate(self, 
                 questions: List[str],
                 answers: List[str], 
                 contexts: List[List[str]],
                 ground_truths: Optional[List[str]] = None) -> RAGASResult:
        """
        Evaluate RAG system outputs using RAGAS metrics.
        
        Args:
            questions: List of input questions
            answers: List of generated answers 
            contexts: List of retrieved contexts for each question
            ground_truths: Optional reference answers for similarity metrics
            
        Returns:
            RAGASResult containing evaluation scores and metadata
        """
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available, returning empty result")
            return RAGASResult(errors=["RAGAS library not available"])
        
        try:
            import time
            start_time = time.time()
            
            # Create RAGAS dataset
            dataset = self._create_ragas_dataset(
                questions, answers, contexts, ground_truths
            )
            
            # Run evaluation
            evaluation_config = self.ragas_config.get('evaluation', {})
            
            if evaluation_config.get('async_evaluation', True):
                # Use async evaluation if available
                result = self._evaluate_async(dataset)
            else:
                # Use synchronous evaluation
                result = self._evaluate_sync(dataset)
            
            evaluation_time = time.time() - start_time
            
            # Process results
            ragas_result = self._process_results(result, evaluation_time)
            
            logger.info(f"RAGAS evaluation completed in {evaluation_time:.2f}s")
            return ragas_result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            return RAGASResult(
                errors=[f"Evaluation failed: {str(e)}"],
                evaluation_time=time.time() - start_time if 'start_time' in locals() else None
            )
    
    def _create_ragas_dataset(self, 
                             questions: List[str],
                             answers: List[str],
                             contexts: List[List[str]], 
                             ground_truths: Optional[List[str]] = None) -> Dataset:
        """Create RAGAS-compatible dataset from inputs."""
        
        # Prepare data dictionary
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts
        }
        
        # Add ground truths if available
        if ground_truths:
            data['ground_truth'] = ground_truths
        
        # Create dataset
        dataset = Dataset.from_dict(data)
        
        # Apply data processing limits
        data_config = self.ragas_config.get('data', {})
        max_examples = self.ragas_config.get('evaluation', {}).get('max_examples')
        
        if max_examples and len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))
            logger.info(f"Limited evaluation to {max_examples} examples")
        
        return dataset
    
    def _evaluate_sync(self, dataset: Dataset) -> Dict[str, Any]:
        """Run synchronous RAGAS evaluation."""
        metrics_list = list(self.enabled_metrics.values())
        
        result = evaluate(
            dataset,
            metrics=metrics_list,
        )
        
        return result
    
    def _evaluate_async(self, dataset: Dataset) -> Dict[str, Any]:
        """Run asynchronous RAGAS evaluation."""
        # For now, fall back to sync evaluation
        # TODO: Implement proper async evaluation when RAGAS supports it
        return self._evaluate_sync(dataset)
    
    def _process_results(self, result: Dict[str, Any], evaluation_time: float) -> RAGASResult:
        """Process raw RAGAS results into structured format."""
        
        # Extract metric scores
        ragas_result = RAGASResult(evaluation_time=evaluation_time)
        
        if 'faithfulness' in result:
            ragas_result.faithfulness = float(result['faithfulness'])
            
        if 'answer_relevancy' in result:
            ragas_result.answer_relevancy = float(result['answer_relevancy'])
            
        if 'context_precision' in result:
            ragas_result.context_precision = float(result['context_precision'])
            
        if 'context_recall' in result:
            ragas_result.context_recall = float(result['context_recall'])
            
        if 'answer_similarity' in result:
            ragas_result.answer_similarity = float(result['answer_similarity'])
            
        if 'answer_correctness' in result:
            ragas_result.answer_correctness = float(result['answer_correctness'])
        
        # Calculate overall score
        scores = []
        if ragas_result.faithfulness is not None:
            scores.append(ragas_result.faithfulness)
        if ragas_result.answer_relevancy is not None:
            scores.append(ragas_result.answer_relevancy)
        if ragas_result.context_precision is not None:
            scores.append(ragas_result.context_precision)
        if ragas_result.context_recall is not None:
            scores.append(ragas_result.context_recall)
            
        if scores:
            ragas_result.overall_score = float(np.mean(scores))
            ragas_result.metric_count = len(scores)
        
        # Add model information
        ragas_result.model_info = {
            'embedding_model': self.embedding_model,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model
        }
        
        return ragas_result
    
    def assess_quality(self, ragas_result: RAGASResult) -> Dict[str, str]:
        """
        Provide qualitative assessment based on RAGAS scores.
        
        Args:
            ragas_result: RAGAS evaluation results
            
        Returns:
            Dictionary with quality assessments for each metric
        """
        if not ragas_result or ragas_result.errors:
            return {'error': 'No valid RAGAS results available'}
        
        thresholds = self.ragas_config.get('thresholds', {})
        assessment = {}
        
        # Assess faithfulness
        if ragas_result.faithfulness is not None:
            score = ragas_result.faithfulness
            if score >= thresholds.get('excellent_faithfulness', 0.8):
                assessment['faithfulness'] = 'Excellent - Highly grounded in context'
            elif score >= thresholds.get('good_faithfulness', 0.6):
                assessment['faithfulness'] = 'Good - Reasonably grounded'
            elif score >= thresholds.get('poor_faithfulness', 0.4):
                assessment['faithfulness'] = 'Fair - Some grounding issues'
            else:
                assessment['faithfulness'] = 'Poor - Poorly grounded in context'
        
        # Assess answer relevancy
        if ragas_result.answer_relevancy is not None:
            score = ragas_result.answer_relevancy
            if score >= thresholds.get('excellent_relevancy', 0.8):
                assessment['answer_relevancy'] = 'Excellent - Highly relevant'
            elif score >= thresholds.get('good_relevancy', 0.6):
                assessment['answer_relevancy'] = 'Good - Moderately relevant'
            elif score >= thresholds.get('poor_relevancy', 0.4):
                assessment['answer_relevancy'] = 'Fair - Somewhat relevant'
            else:
                assessment['answer_relevancy'] = 'Poor - Not very relevant'
        
        # Assess context precision
        if ragas_result.context_precision is not None:
            score = ragas_result.context_precision
            if score >= thresholds.get('excellent_precision', 0.8):
                assessment['context_precision'] = 'Excellent - High signal-to-noise ratio'
            elif score >= thresholds.get('good_precision', 0.6):
                assessment['context_precision'] = 'Good - Moderate precision'
            elif score >= thresholds.get('poor_precision', 0.4):
                assessment['context_precision'] = 'Fair - Some noise in contexts'
            else:
                assessment['context_precision'] = 'Poor - Low precision'
        
        # Assess context recall
        if ragas_result.context_recall is not None:
            score = ragas_result.context_recall
            if score >= thresholds.get('excellent_recall', 0.8):
                assessment['context_recall'] = 'Excellent - Captures most relevant information'
            elif score >= thresholds.get('good_recall', 0.6):
                assessment['context_recall'] = 'Good - Captures some relevant information'
            elif score >= thresholds.get('poor_recall', 0.4):
                assessment['context_recall'] = 'Fair - Misses some important information'
            else:
                assessment['context_recall'] = 'Poor - Misses critical information'
        
        # Overall assessment
        if ragas_result.overall_score is not None:
            score = ragas_result.overall_score
            if score >= 0.8:
                assessment['overall'] = 'Excellent RAG system performance'
            elif score >= 0.6:
                assessment['overall'] = 'Good RAG system performance'
            elif score >= 0.4:
                assessment['overall'] = 'Fair RAG system performance'
            else:
                assessment['overall'] = 'Poor RAG system performance'
        
        return assessment
    
    def export_results(self, ragas_result: RAGASResult, output_path: str):
        """
        Export RAGAS results to file.
        
        Args:
            ragas_result: RAGAS evaluation results
            output_path: Path to save results
        """
        try:
            # Convert to serializable format
            results_dict = {
                'ragas_metrics': {
                    'faithfulness': ragas_result.faithfulness,
                    'answer_relevancy': ragas_result.answer_relevancy,
                    'context_precision': ragas_result.context_precision,
                    'context_recall': ragas_result.context_recall,
                    'answer_similarity': ragas_result.answer_similarity,
                    'answer_correctness': ragas_result.answer_correctness,
                    'overall_score': ragas_result.overall_score,
                    'metric_count': ragas_result.metric_count
                },
                'metadata': {
                    'evaluation_time': ragas_result.evaluation_time,
                    'model_info': ragas_result.model_info,
                    'errors': ragas_result.errors
                },
                'quality_assessment': self.assess_quality(ragas_result)
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"RAGAS results exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export RAGAS results: {str(e)}")


def create_ragas_evaluator(config: Dict[str, Any]) -> Optional[RAGASEvaluator]:
    """
    Factory function to create RAGAS evaluator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RAGASEvaluator instance or None if disabled/unavailable
    """
    ragas_config = config.get('ragas', {})
    
    if not ragas_config.get('enabled', False):
        logger.info("RAGAS evaluation disabled in configuration")
        return None
    
    if not RAGAS_AVAILABLE and not ragas_config.get('fallback_on_error', True):
        logger.error("RAGAS library not available and fallback disabled")
        return None
    
    try:
        return RAGASEvaluator(config)
    except Exception as e:
        logger.error(f"Failed to create RAGAS evaluator: {str(e)}")
        return None