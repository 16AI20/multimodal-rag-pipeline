"""
RAG System Evaluator
Evaluates using ChromaDB metrics, BLEU/ROUGE, and ChatGPT comparison
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import BLEU/ROUGE (install with: pip install nltk rouge-score)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: NLTK and rouge-score not installed. Install with: pip install nltk rouge-score")
    METRICS_AVAILABLE = False

# Import RAG pipeline
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.rag.pipeline import RAGPipeline

# Import RAGAS evaluator
from .ragas_evaluator import create_ragas_evaluator, RAGASResult
from .data_adapters import RAGPipelineToRAGAS, EvaluationDataLoader, RAGASResultsProcessor


class RAGEvaluator:
    """Comprehensive RAG system evaluator."""
    
    def __init__(self, rag_pipeline: RAGPipeline, config_path: str = None) -> None:
        """Initialize the RAG evaluator.
        
        Args:
            rag_pipeline: RAG pipeline instance to evaluate.
            config_path: Path to configuration directory.
        """
        self.rag_pipeline = rag_pipeline
        self.config_path = config_path or "conf"
        
        # Load evaluation configuration
        from ..utils import load_config
        self.config = load_config(self.config_path)
        self.eval_config = self.config.get('evaluation', {})
        
        # Initialize semantic similarity model for comparisons
        similarity_model_name = self.eval_config.get('similarity_model', 'all-MiniLM-L6-v2')
        self.similarity_model = SentenceTransformer(similarity_model_name)
        
        # Initialize ROUGE scorer
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
        
        # Initialize RAGAS evaluator
        self.ragas_evaluator = create_ragas_evaluator(self.config)
        self.data_adapter = RAGPipelineToRAGAS(self.config) if self.ragas_evaluator else None
        self.results_processor = RAGASResultsProcessor(self.config) if self.ragas_evaluator else None
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_chromadb_metrics(self, questions: List[str]) -> Dict[str, Any]:
        """Evaluate using ChromaDB built-in retrieval metrics.
        
        Args:
            questions: List of test questions to evaluate.
            
        Returns:
            Dictionary containing individual and aggregate metrics.
        """
        results = []
        
        for question in questions:
            try:
                result = self.rag_pipeline.query(question, k=5)
                
                # Check if query was successful based on response content
                if result.get('answer') and result.get('answer') != "I couldn't find any relevant information to answer your question.":
                    # Extract sources information 
                    sources = result.get('sources', [])
                    
                    retrieval_metrics = {
                        'question': question,
                        'retrieval_confidence': result.get('retrieval_metadata', {}).get('confidence_score', 0.0),
                        'num_sources': len(sources),
                        'unique_sources': len(set([s.get('citation', s.get('source', '')) for s in sources])),
                        'avg_similarity': np.mean([s.get('score', s.get('similarity_score', 0)) for s in sources]) if sources else 0,
                        'max_similarity': max([s.get('score', s.get('similarity_score', 0)) for s in sources]) if sources else 0,
                        'source_diversity': len(set([s.get('file_type', 'unknown') for s in sources])),
                        'response_length': len(result['answer'].split()),
                        'success': True
                    }
                else:
                    retrieval_metrics = {
                        'question': question,
                        'error': result.get('error', 'No relevant information found'),
                        'success': False
                    }
                
                results.append(retrieval_metrics)
                
            except Exception as e:
                self.logger.error(f"Error evaluating question '{question}': {e}")
                results.append({
                    'question': question,
                    'error': str(e),
                    'success': False
                })
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r.get('success', False)]
        
        aggregate_metrics = {
            'total_questions': len(questions),
            'successful_queries': len(successful_results),
            'success_rate': len(successful_results) / len(questions),
            'avg_retrieval_confidence': np.mean([r['retrieval_confidence'] for r in successful_results if 'retrieval_confidence' in r]),
            'avg_num_sources': np.mean([r['num_sources'] for r in successful_results if 'num_sources' in r]),
            'avg_source_diversity': np.mean([r['source_diversity'] for r in successful_results if 'source_diversity' in r]),
            'avg_response_length': np.mean([r['response_length'] for r in successful_results if 'response_length' in r])
        }
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def evaluate_bleu_rouge(self, evaluation_file: str) -> Dict[str, Any]:
        """Evaluate using BLEU and ROUGE metrics against reference answers.
        
        Args:
            evaluation_file: Path to JSON file with questions and reference answers.
            
        Returns:
            Dictionary containing BLEU/ROUGE scores and comparisons.
        """
        if not METRICS_AVAILABLE:
            return {'error': 'NLTK and rouge-score packages not installed'}
        
        # Load evaluation data
        with open(evaluation_file, 'r') as f:
            eval_data = json.load(f)
        
        results = []
        
        for item in eval_data['questions']:
            question = item['question']
            reference_answer = item['reference_answer']
            
            if reference_answer == "FILL_IN_HERE":
                self.logger.info(f"Skipping unfilled question: {question}")
                continue  # Skip unfilled template questions
            
            try:
                # Get RAG system response
                self.logger.info(f"Evaluating BLEU/ROUGE for: {question}")
                rag_result = self.rag_pipeline.query(question)
                self.logger.debug(f"RAG result keys: {rag_result.keys()}")
                
                if 'error' not in rag_result and rag_result.get('answer'):
                    generated_answer = rag_result['answer']
                    self.logger.info(f"Generated answer length: {len(generated_answer.split())} words")
                    
                    # Calculate BLEU score
                    reference_tokens = reference_answer.split()
                    generated_tokens = generated_answer.split()
                    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=self.smoothing)
                    
                    # Calculate ROUGE scores
                    rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
                    
                    # Calculate semantic similarity
                    ref_embedding = self.similarity_model.encode([reference_answer])
                    gen_embedding = self.similarity_model.encode([generated_answer])
                    semantic_similarity = cosine_similarity(ref_embedding, gen_embedding)[0][0]
                    
                    result = {
                        'question': question,
                        'category': item['category'],
                        'reference_answer': reference_answer,
                        'generated_answer': generated_answer,
                        'bleu_score': bleu_score,
                        'rouge1_f1': rouge_scores['rouge1'].fmeasure,
                        'rouge2_f1': rouge_scores['rouge2'].fmeasure,
                        'rougeL_f1': rouge_scores['rougeL'].fmeasure,
                        'semantic_similarity': semantic_similarity,
                        'reference_length': len(reference_tokens),
                        'generated_length': len(generated_tokens),
                        'success': True
                    }
                else:
                    self.logger.warning(f"RAG query failed for '{question}': {rag_result.get('error', 'No answer generated')}")
                    result = {
                        'question': question,
                        'category': item['category'],
                        'error': rag_result.get('error', 'Query failed'),
                        'success': False
                    }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in BLEU/ROUGE evaluation for '{question}': {e}")
                results.append({
                    'question': question,
                    'category': item['category'],
                    'error': str(e),
                    'success': False
                })
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r.get('success', False)]
        
        self.logger.info(f"BLEU/ROUGE evaluation: {len(successful_results)}/{len(results)} successful")
        
        if successful_results:
            aggregate_metrics = {
                'avg_bleu_score': np.mean([r['bleu_score'] for r in successful_results]),
                'avg_rouge1_f1': np.mean([r['rouge1_f1'] for r in successful_results]),
                'avg_rouge2_f1': np.mean([r['rouge2_f1'] for r in successful_results]),
                'avg_rougeL_f1': np.mean([r['rougeL_f1'] for r in successful_results]),
                'avg_semantic_similarity': np.mean([r['semantic_similarity'] for r in successful_results]),
                'total_evaluated': len(successful_results)
            }
        else:
            error_reasons = [r.get('error', 'Unknown') for r in results if not r.get('success', False)]
            self.logger.warning(f"BLEU/ROUGE evaluation failed - errors: {set(error_reasons)}")
            aggregate_metrics = {'error': 'No successful evaluations'}
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def evaluate_chatgpt_comparison(self, comparison_file: str) -> Dict[str, Any]:
        """Compare RAG responses with ChatGPT responses.
        
        Args:
            comparison_file: Path to JSON file with questions and ChatGPT answers.
            
        Returns:
            Dictionary containing similarity metrics and comparisons.
        """
        # Load comparison data
        with open(comparison_file, 'r') as f:
            comparison_data = json.load(f)
        
        results = []
        
        for item in comparison_data['questions']:
            question = item['question']
            chatgpt_answer = item['chatgpt_answer']
            
            if chatgpt_answer == "FILL_IN_HERE":
                self.logger.info(f"Skipping unfilled ChatGPT question: {question}")
                continue  # Skip unfilled template questions
            
            try:
                # Get RAG system response
                self.logger.info(f"Evaluating ChatGPT comparison for: {question}")
                rag_result = self.rag_pipeline.query(question)
                
                if 'error' not in rag_result and rag_result.get('answer'):
                    rag_answer = rag_result['answer']
                    self.logger.info(f"RAG answer vs ChatGPT: {len(rag_answer.split())} vs {len(chatgpt_answer.split())} words")
                    
                    # Calculate semantic similarity between responses
                    rag_embedding = self.similarity_model.encode([rag_answer])
                    chatgpt_embedding = self.similarity_model.encode([chatgpt_answer])
                    response_similarity = cosine_similarity(rag_embedding, chatgpt_embedding)[0][0]
                    
                    # Length comparison
                    rag_length = len(rag_answer.split())
                    chatgpt_length = len(chatgpt_answer.split())
                    
                    result = {
                        'question': question,
                        'rag_answer': rag_answer,
                        'chatgpt_answer': chatgpt_answer,
                        'response_similarity': response_similarity,
                        'rag_length': rag_length,
                        'chatgpt_length': chatgpt_length,
                        'length_difference': abs(rag_length - chatgpt_length),
                        'rag_sources': len(rag_result['sources']),
                        'rag_confidence': rag_result['retrieval_metadata']['confidence_score'],
                        'success': True
                    }
                else:
                    result = {
                        'question': question,
                        'chatgpt_answer': chatgpt_answer,
                        'rag_error': rag_result.get('error', 'Query failed'),
                        'success': False
                    }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in ChatGPT comparison for '{question}': {e}")
                results.append({
                    'question': question,
                    'error': str(e),
                    'success': False
                })
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r.get('success', False)]
        
        self.logger.info(f"ChatGPT comparison: {len(successful_results)}/{len(results)} successful")
        
        if successful_results:
            aggregate_metrics = {
                'avg_response_similarity': np.mean([r['response_similarity'] for r in successful_results]),
                'avg_rag_length': np.mean([r['rag_length'] for r in successful_results]),
                'avg_chatgpt_length': np.mean([r['chatgpt_length'] for r in successful_results]),
                'avg_length_difference': np.mean([r['length_difference'] for r in successful_results]),
                'total_compared': len(successful_results)
            }
        else:
            error_reasons = [r.get('error', 'Unknown') for r in results if not r.get('success', False)]
            self.logger.warning(f"ChatGPT comparison failed - errors: {set(error_reasons)}")
            aggregate_metrics = {'error': 'No successful comparisons'}
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def evaluate_ragas_metrics(self, questions: List[str], ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            questions: List of test questions
            ground_truths: Optional reference answers for similarity metrics
            
        Returns:
            Dictionary containing RAGAS evaluation results
        """
        if not self.ragas_evaluator or not self.data_adapter:
            self.logger.warning("RAGAS evaluator not available, skipping RAGAS evaluation")
            return {
                'error': 'RAGAS evaluator not available',
                'available': False
            }
        
        try:
            self.logger.info(f"Starting RAGAS evaluation with {len(questions)} questions")
            
            # Generate RAG responses for all questions
            rag_responses = []
            for question in questions:
                try:
                    response = self.rag_pipeline.query(question, k=5)
                    rag_responses.append(response)
                except Exception as e:
                    self.logger.error(f"Failed to get RAG response for '{question}': {str(e)}")
                    # Create minimal response for failed queries
                    rag_responses.append({
                        'query': question,
                        'answer': 'Failed to generate response',
                        'sources': [],
                        'error': str(e)
                    })
            
            # Convert to RAGAS format
            ragas_batch_data = self.data_adapter.convert_batch_responses(
                rag_responses, ground_truths
            )
            
            # Extract data for RAGAS evaluation
            questions_list = ragas_batch_data['question']
            answers_list = ragas_batch_data['answer'] 
            contexts_list = ragas_batch_data['contexts']
            ground_truth_list = ragas_batch_data.get('ground_truth')
            
            # Run RAGAS evaluation
            ragas_result = self.ragas_evaluator.evaluate(
                questions_list, answers_list, contexts_list, ground_truth_list
            )
            
            # Get quality assessment
            quality_assessment = self.ragas_evaluator.assess_quality(ragas_result)
            
            # Build results dictionary
            results = {
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
                'quality_assessment': quality_assessment,
                'metadata': {
                    'evaluation_time': ragas_result.evaluation_time,
                    'model_info': ragas_result.model_info,
                    'questions_count': len(questions),
                    'successful_responses': len([r for r in rag_responses if 'error' not in r])
                },
                'available': True
            }
            
            # Add errors if any
            if ragas_result.errors:
                results['errors'] = ragas_result.errors
            
            self.logger.info(f"RAGAS evaluation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"RAGAS evaluation failed: {str(e)}")
            return {
                'error': f'RAGAS evaluation failed: {str(e)}',
                'available': False
            }
    
    def generate_markdown_report(self, results: Dict[str, Any], output_file: str) -> None:
        """Generate a Markdown report from evaluation results.
        
        Args:
            results: Evaluation results dictionary.
            output_file: Path to output Markdown file.
        """
        from datetime import datetime
        
        report = []
        report.append("# RAG System Evaluation Report")
        report.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # ChromaDB Metrics Section
        if 'chromadb_metrics' in results:
            chromadb = results['chromadb_metrics']['aggregate_metrics']
            report.append("## 📊 ChromaDB Retrieval Metrics")
            report.append("")
            report.append("**Methodology:** This evaluation tests the RAG system's ability to retrieve relevant documents from the ChromaDB vector store. We query the system with 5 standard questions and measure retrieval quality, source diversity, and response characteristics. Each query requests the top 5 most similar documents based on embedding similarity scores.")
            report.append("")
            report.append("| Metric | Value | Description |")
            report.append("|--------|-------|-------------|")
            report.append(f"| Success Rate | {chromadb['success_rate']:.2%} | Percentage of queries that returned meaningful results |")
            report.append(f"| Avg Retrieval Confidence | {chromadb['avg_retrieval_confidence']:.3f} | Average similarity score between query and retrieved documents |")
            report.append(f"| Avg Sources Retrieved | {chromadb['avg_num_sources']:.1f} | Average number of relevant documents found per query |")
            report.append(f"| Avg Source Diversity | {chromadb['avg_source_diversity']:.1f} | Average number of different document types retrieved |")
            report.append(f"| Avg Response Length | {chromadb['avg_response_length']:.1f} words | Average length of generated responses |\n")
        
        # BLEU/ROUGE Section
        if 'bleu_rouge' in results and 'aggregate_metrics' in results['bleu_rouge']:
            bleu_rouge = results['bleu_rouge']['aggregate_metrics']
            
            # Check if evaluation was successful (has metrics, not just error)
            if 'error' not in bleu_rouge and 'avg_bleu_score' in bleu_rouge:
                report.append("## 📝 BLEU/ROUGE Scores vs Reference Answers")
                report.append("")
                report.append("**Methodology:** This evaluation compares RAG-generated answers against human-written reference answers using established NLP metrics. BLEU measures n-gram overlap (exact word/phrase matching), ROUGE measures recall-oriented overlap at different granularities, and semantic similarity uses BERT embeddings to capture meaning-based similarity beyond exact word matching.")
                report.append("")
                report.append("| Metric | Score | Description | Interpretation |")
                report.append("|--------|-------|-------------|---------------|")
                report.append(f"| BLEU Score | {bleu_rouge['avg_bleu_score']:.3f} | N-gram overlap with reference | Measures exact word/phrase matching |")
                report.append(f"| ROUGE-1 F1 | {bleu_rouge['avg_rouge1_f1']:.3f} | Unigram overlap | Individual word overlap |")
                report.append(f"| ROUGE-2 F1 | {bleu_rouge['avg_rouge2_f1']:.3f} | Bigram overlap | Two-word sequence overlap |")
                report.append(f"| ROUGE-L F1 | {bleu_rouge['avg_rougeL_f1']:.3f} | Longest common subsequence | Preserves word order |")
                report.append(f"| Semantic Similarity | {bleu_rouge['avg_semantic_similarity']:.3f} | BERT-based similarity | Meaning-based comparison |\n")
                
                # Individual question results
                report.append("### Detailed Results vs Human-Provided Reference Answers")
                successful_individual = [r for r in results['bleu_rouge']['individual_results'] if r.get('success', False)]
                
                if successful_individual:
                    for result in successful_individual:
                        report.append(f"#### {result['question']}")
                        report.append("")
                        report.append("**Human Reference Answer:**")
                        # Handle multi-paragraph blockquotes
                        ref_lines = result['reference_answer'].strip().split('\n')
                        for line in ref_lines:
                            if line.strip():
                                report.append(f"> {line}")
                            else:
                                report.append(">")
                        report.append("")
                        report.append("**RAG System Generated Answer:**")
                        # Handle multi-paragraph blockquotes
                        gen_lines = result['generated_answer'].strip().split('\n')
                        for line in gen_lines:
                            if line.strip():
                                report.append(f"> {line}")
                            else:
                                report.append(">")
                        report.append("")
                        report.append("**Evaluation Metrics:**")
                        report.append("| Metric | Score |")
                        report.append("|--------|-------|")
                        report.append(f"| BLEU Score | {result['bleu_score']:.3f} |")
                        report.append(f"| ROUGE-L F1 | {result['rougeL_f1']:.3f} |")
                        report.append(f"| Semantic Similarity | {result['semantic_similarity']:.3f} |")
                        report.append("")
                else:
                    report.append("No successful individual evaluations to display.")
                    report.append("")
            else:
                report.append("## 📝 BLEU/ROUGE Scores vs Reference Answers")
                report.append("")
                report.append("**Methodology:** This evaluation compares RAG-generated answers against human-written reference answers using established NLP metrics. BLEU measures n-gram overlap (exact word/phrase matching), ROUGE measures recall-oriented overlap at different granularities, and semantic similarity uses BERT embeddings to capture meaning-based similarity beyond exact word matching.")
                report.append("")
                report.append("❌ **BLEU/ROUGE evaluation failed or no successful evaluations**")
                if 'error' in bleu_rouge:
                    report.append(f"Error: {bleu_rouge['error']}")
                report.append("")
        
        # ChatGPT Comparison Section
        if 'chatgpt_comparison' in results and 'aggregate_metrics' in results['chatgpt_comparison']:
            chatgpt = results['chatgpt_comparison']['aggregate_metrics']
            
            # Check if evaluation was successful
            if 'error' not in chatgpt and 'avg_response_similarity' in chatgpt:
                report.append("## 🤖 RAG vs ChatGPT Comparison")
                report.append("")
                report.append("**Methodology:** This evaluation compares the RAG system's responses against ChatGPT's responses to the same questions. We measure semantic similarity using BERT embeddings to assess how closely the responses align in meaning, and analyze response length characteristics. This helps evaluate consistency and identifies areas where the RAG system differs from a general-purpose LLM.")
                report.append("")
                report.append("| Metric | Value |")
                report.append("|--------|-------|")
                report.append(f"| Avg Response Similarity | {chatgpt['avg_response_similarity']:.3f} |")
                report.append(f"| Avg RAG Length | {chatgpt['avg_rag_length']:.1f} words |")
                report.append(f"| Avg ChatGPT Length | {chatgpt['avg_chatgpt_length']:.1f} words |")
                report.append(f"| Avg Length Difference | {chatgpt['avg_length_difference']:.1f} words |\n")
                
                # Individual comparisons
                report.append("### RAG vs ChatGPT Response Comparison")
                successful_chatgpt = [r for r in results['chatgpt_comparison']['individual_results'] if r.get('success', False)]
                
                if successful_chatgpt:
                    for result in successful_chatgpt:
                        report.append(f"#### {result['question']}")
                        report.append("")
                        report.append("**RAG System Generated Answer:**")
                        # Handle multi-paragraph blockquotes
                        rag_lines = result['rag_answer'].strip().split('\n')
                        for line in rag_lines:
                            if line.strip():
                                report.append(f"> {line}")
                            else:
                                report.append(">")
                        report.append("")
                        report.append("**ChatGPT Generated Answer:**")
                        # Handle multi-paragraph blockquotes
                        chatgpt_lines = result['chatgpt_answer'].strip().split('\n')
                        for line in chatgpt_lines:
                            if line.strip():
                                report.append(f"> {line}")
                            else:
                                report.append(">")
                        report.append("")
                        report.append("**Evaluation Metrics:**")
                        report.append("| Metric | Score |")
                        report.append("|--------|-------|")
                        report.append(f"| Response Similarity | {result['response_similarity']:.3f} |")
                        report.append(f"| RAG Length | {result['rag_length']} words |")
                        report.append(f"| ChatGPT Length | {result['chatgpt_length']} words |")
                        report.append("")
                else:
                    report.append("No successful individual comparisons to display.")
                    report.append("")
            else:
                report.append("## 🤖 RAG vs ChatGPT Comparison")
                report.append("")
                report.append("**Methodology:** This evaluation compares the RAG system's responses against ChatGPT's responses to the same questions. We measure semantic similarity using BERT embeddings to assess how closely the responses align in meaning, and analyze response length characteristics. This helps evaluate consistency and identifies areas where the RAG system differs from a general-purpose LLM.")
                report.append("")
                report.append("❌ **ChatGPT comparison failed or no successful comparisons**")
                if 'error' in chatgpt:
                    report.append(f"Error: {chatgpt['error']}")
                report.append("")
        
        # Evaluation Summary
        report.append("## 📈 Evaluation Summary")
        
        # Metrics Overview
        report.append("### 📋 Overall Metrics")
        report.append("")
        report.append("| Evaluation Component | Key Metrics | Performance |")
        report.append("|---------------------|-------------|-------------|")
        
        if 'chromadb_metrics' in results:
            chromadb_summary = results['chromadb_metrics']['aggregate_metrics']
            report.append(f"| **ChromaDB Retrieval** | Success Rate, Confidence | {chromadb_summary['success_rate']:.1%}, {chromadb_summary['avg_retrieval_confidence']:.3f} |")
        
        if 'bleu_rouge' in results and 'aggregate_metrics' in results['bleu_rouge']:
            bleu_summary = results['bleu_rouge']['aggregate_metrics']
            if 'error' not in bleu_summary:
                report.append(f"| **BLEU/ROUGE vs References** | BLEU, ROUGE-L, Semantic | {bleu_summary['avg_bleu_score']:.3f}, {bleu_summary['avg_rougeL_f1']:.3f}, {bleu_summary['avg_semantic_similarity']:.3f} |")
            else:
                report.append("| **BLEU/ROUGE vs References** | BLEU, ROUGE-L, Semantic | ❌ Evaluation failed |")
        
        if 'chatgpt_comparison' in results and 'aggregate_metrics' in results['chatgpt_comparison']:
            chatgpt_summary = results['chatgpt_comparison']['aggregate_metrics']
            if 'error' not in chatgpt_summary:
                report.append(f"| **RAG vs ChatGPT** | Similarity, Length Diff | {chatgpt_summary['avg_response_similarity']:.3f}, {abs(chatgpt_summary['avg_rag_length'] - chatgpt_summary['avg_chatgpt_length']):.1f} words |")
            else:
                report.append("| **RAG vs ChatGPT** | Similarity, Length Diff | ❌ Comparison failed |")
        
        report.append("")
        
        # Recommendations
        report.append("### 🎯 Recommendations")
        
        if 'chromadb_metrics' in results:
            conf = results['chromadb_metrics']['aggregate_metrics']['avg_retrieval_confidence']
            success_rate = results['chromadb_metrics']['aggregate_metrics']['success_rate']
            avg_sources = results['chromadb_metrics']['aggregate_metrics']['avg_num_sources']
            
            if not np.isnan(conf):
                if conf > 0.8:
                    report.append("- ✅ **Retrieval Quality**: Excellent performance with high confidence scores (>0.8). The system is effectively matching user queries to relevant documents.")
                    report.append("  - *Current strength*: Strong semantic understanding between queries and corpus content")
                    report.append("  - *Maintenance*: Monitor for any drift in performance over time as corpus grows")
                elif conf > 0.6:
                    report.append(f"- ⚠️ **Retrieval Quality**: Good performance (confidence: {conf:.3f}) but room for improvement.")
                    report.append("  - *Recommended actions*: Experiment with different embedding models (e.g., try larger BGE models or domain-specific embeddings)")
                    report.append("  - *Configuration tuning*: Adjust chunk size (current default ~1000 chars) - try 800-1200 char ranges")
                    report.append("  - *Content optimization*: Review document preprocessing to ensure clean, meaningful chunks")
                else:
                    report.append(f"- ❌ **Retrieval Quality**: Low confidence scores ({conf:.3f}) indicate poor semantic matching.")
                    report.append("  - *Immediate actions*: Review corpus quality - check for corrupted/empty documents")
                    report.append("  - *Embedding strategy*: Consider domain-specific embedding models trained on similar content")
                    report.append("  - *Vector store*: Rebuild embeddings with updated preprocessing pipeline")
                    report.append("  - *Evaluation*: Manually test queries to identify systematic retrieval issues")
            else:
                report.append("- ❌ **Retrieval Quality**: System failure - no successful document retrievals.")
                report.append("  - *Critical check*: Verify ChromaDB connection and vector store exists")
                report.append("  - *Configuration*: Review embedding model loading and device compatibility")
                report.append("  - *Debugging*: Check logs for specific error messages during retrieval")
        
        if 'bleu_rouge' in results and 'aggregate_metrics' in results['bleu_rouge']:
            bleu_rouge_agg = results['bleu_rouge']['aggregate_metrics']
            if 'error' not in bleu_rouge_agg and 'avg_rougeL_f1' in bleu_rouge_agg:
                rouge_l = bleu_rouge_agg['avg_rougeL_f1']
                bleu_score = bleu_rouge_agg['avg_bleu_score']
                semantic_sim = bleu_rouge_agg['avg_semantic_similarity']
                
                if not np.isnan(rouge_l):
                    if rouge_l > 0.5:
                        report.append(f"- ✅ **Answer Quality**: Excellent alignment with reference answers (ROUGE-L: {rouge_l:.3f}, Semantic: {semantic_sim:.3f}).")
                        report.append("  - *Current strength*: Responses closely match human-written references in both structure and meaning")
                        report.append("  - *Optimization*: Fine-tune response length and detail level for different question types")
                    elif rouge_l > 0.3:
                        report.append(f"- ⚠️ **Answer Quality**: Moderate alignment (ROUGE-L: {rouge_l:.3f}, BLEU: {bleu_score:.3f}) with room for improvement.")
                        report.append("  - *Prompt engineering*: Refine system prompts to better match reference answer style and structure")
                        report.append("  - *Retrieval improvement*: Ensure top-k retrieval captures the most relevant context for accurate answers")
                        report.append("  - *LLM parameters*: Experiment with temperature and max_tokens settings for more precise responses")
                    else:
                        report.append(f"- ❌ **Answer Quality**: Low overlap with references (ROUGE-L: {rouge_l:.3f}) suggests significant issues.")
                        report.append("  - *Critical review*: Examine LLM configuration - check model, temperature, and prompt templates")
                        report.append("  - *Content analysis*: Verify retrieval is finding relevant documents for each question")
                        report.append("  - *Reference validation*: Ensure reference answers are appropriate and achievable targets")
                        report.append("  - *Model evaluation*: Consider testing different LLM models or fine-tuning approaches")
                else:
                    report.append("- ❌ **Answer Quality**: Evaluation failed to produce valid metrics.")
                    report.append("  - *Data check*: Verify evaluation questions have proper reference answers")
                    report.append("  - *Pipeline validation*: Test RAG system responses manually for basic functionality")
            else:
                report.append("- ❌ **Answer Quality**: BLEU/ROUGE evaluation failed - system or configuration issue.")
                report.append("  - *Dependencies*: Ensure NLTK and rouge-score packages are properly installed")
                report.append("  - *File validation*: Check evaluation data files exist and have correct format")
                report.append("  - *Error analysis*: Review evaluation logs for specific failure reasons")
        
        if 'chatgpt_comparison' in results and 'aggregate_metrics' in results['chatgpt_comparison']:
            chatgpt_agg = results['chatgpt_comparison']['aggregate_metrics']
            if 'error' not in chatgpt_agg and 'avg_response_similarity' in chatgpt_agg:
                similarity = chatgpt_agg['avg_response_similarity']
                avg_rag_length = chatgpt_agg['avg_rag_length']
                avg_chatgpt_length = chatgpt_agg['avg_chatgpt_length']
                length_diff = abs(avg_rag_length - avg_chatgpt_length)
                
                if not np.isnan(similarity):
                    if similarity > 0.7:
                        report.append(f"- ✅ **Response Consistency**: High alignment with ChatGPT (similarity: {similarity:.3f}, length diff: {length_diff:.1f} words).")
                        report.append("  - *Current strength*: RAG responses are semantically consistent with general AI assistant responses")
                        report.append("  - *Advantage*: System provides domain-specific accuracy while maintaining natural response style")
                        report.append("  - *Monitoring*: Track consistency over time as prompts or models change")
                    elif similarity > 0.5:
                        report.append(f"- ⚠️ **Response Consistency**: Moderate alignment (similarity: {similarity:.3f}) - responses differ but may be valid.")
                        report.append("  - *Analysis needed*: Review individual comparisons to understand differences")
                        report.append("  - *Domain specialization*: Differences may indicate domain-specific knowledge (positive)")
                        report.append("  - *Style alignment*: Consider adjusting prompts if response style consistency is important")
                        report.append(f"  - *Length consideration*: Average length difference of {length_diff:.1f} words may indicate verbosity misalignment")
                    else:
                        report.append(f"- ❌ **Response Consistency**: Low similarity ({similarity:.3f}) suggests significant divergence from expected responses.")
                        report.append("  - *Immediate review*: Examine individual response comparisons to identify systematic issues")
                        report.append("  - *Prompt alignment*: Ensure system prompts encourage similar response style and tone as desired")
                        report.append("  - *Model behavior*: Investigate if LLM configuration is producing unexpected response patterns")
                        report.append(f"  - *Length analysis*: Large length difference ({length_diff:.1f} words) may indicate over/under-generation issues")
                else:
                    report.append("- ❌ **Response Consistency**: Comparison metrics failed to calculate.")
                    report.append("  - *Data validation*: Check that both RAG and ChatGPT responses exist for comparison questions")
                    report.append("  - *Model access*: Verify semantic similarity model (BERT) is loading correctly")
            else:
                report.append("- ❌ **Response Consistency**: ChatGPT comparison evaluation failed.")
                report.append("  - *File check*: Ensure ChatGPT comparison answers file exists and is properly formatted")
                report.append("  - *Dependencies*: Verify sentence-transformers package is installed for similarity calculation")
                report.append("  - *Error diagnosis*: Check evaluation logs for specific comparison failures")
        
        # RAGAS Evaluation Section
        if 'ragas' in results:
            ragas_results = results['ragas']
            
            if ragas_results.get('available', False) and 'ragas_metrics' in ragas_results:
                ragas_metrics = ragas_results['ragas_metrics']
                quality_assessment = ragas_results.get('quality_assessment', {})
                
                report.append("")
                report.append("## 🎯 RAGAS (RAG Assessment) Metrics")
                report.append("")
                report.append("**Methodology:** RAGAS provides specialized metrics designed specifically for RAG systems. Faithfulness measures how grounded responses are in retrieved context, Answer Relevancy evaluates how well answers address questions, Context Precision measures signal-to-noise ratio in retrieved documents, and Context Recall assesses retrieval completeness.")
                report.append("")
                
                # Overall RAGAS score
                overall_score = ragas_metrics.get('overall_score')
                if overall_score is not None:
                    report.append(f"**Overall RAGAS Score: {overall_score:.3f}**")
                    report.append("")
                
                # Individual metrics table
                report.append("| RAGAS Metric | Score | Assessment |")
                report.append("|--------------|-------|------------|")
                
                # Faithfulness
                faithfulness = ragas_metrics.get('faithfulness')
                if faithfulness is not None:
                    faithfulness_assessment = quality_assessment.get('faithfulness', 'Not assessed')
                    report.append(f"| Faithfulness | {faithfulness:.3f} | {faithfulness_assessment} |")
                
                # Answer Relevancy
                answer_relevancy = ragas_metrics.get('answer_relevancy')
                if answer_relevancy is not None:
                    relevancy_assessment = quality_assessment.get('answer_relevancy', 'Not assessed')
                    report.append(f"| Answer Relevancy | {answer_relevancy:.3f} | {relevancy_assessment} |")
                
                # Context Precision
                context_precision = ragas_metrics.get('context_precision')
                if context_precision is not None:
                    precision_assessment = quality_assessment.get('context_precision', 'Not assessed')
                    report.append(f"| Context Precision | {context_precision:.3f} | {precision_assessment} |")
                
                # Context Recall
                context_recall = ragas_metrics.get('context_recall')
                if context_recall is not None:
                    recall_assessment = quality_assessment.get('context_recall', 'Not assessed')
                    report.append(f"| Context Recall | {context_recall:.3f} | {recall_assessment} |")
                
                report.append("")
                
                # Overall assessment
                overall_assessment = quality_assessment.get('overall', 'No overall assessment available')
                if overall_score is not None:
                    if overall_score >= 0.8:
                        report.append("- ✅ **RAGAS Assessment**: Excellent RAG system performance across all dimensions.")
                        report.append("  - *Strengths*: High faithfulness and relevancy indicate reliable, grounded responses")
                        report.append("  - *Maintenance*: Monitor performance consistency as corpus or models change")
                    elif overall_score >= 0.6:
                        report.append("- ⚠️ **RAGAS Assessment**: Good performance with opportunities for improvement.")
                        report.append("  - *Focus areas*: Review lower-scoring metrics for targeted improvements")
                        report.append("  - *Optimization*: Consider prompt engineering, retrieval tuning, or model adjustments")
                    elif overall_score >= 0.4:
                        report.append("- ⚠️ **RAGAS Assessment**: Fair performance requiring attention to multiple areas.")
                        report.append("  - *Priority*: Address faithfulness issues if present to prevent hallucination")
                        report.append("  - *Retrieval*: Improve context precision and recall through better document processing")
                        report.append("  - *Generation*: Enhance answer relevancy through prompt optimization")
                    else:
                        report.append("- ❌ **RAGAS Assessment**: Poor performance requiring systematic improvements.")
                        report.append("  - *Critical*: Low scores indicate fundamental issues with RAG pipeline")
                        report.append("  - *Immediate actions*: Review and rebuild core components (retrieval, generation)")
                        report.append("  - *Validation*: Test individual pipeline stages to identify bottlenecks")
                
                # Add specific recommendations based on individual metrics
                if faithfulness is not None and faithfulness < 0.5:
                    report.append("  - *Faithfulness concern*: High risk of hallucinated responses - review prompt templates and context usage")
                
                if answer_relevancy is not None and answer_relevancy < 0.5:
                    report.append("  - *Relevancy concern*: Responses may not address questions properly - review question understanding and response generation")
                
                if context_precision is not None and context_precision < 0.5:
                    report.append("  - *Precision concern*: Too much noise in retrieved contexts - improve document chunking and retrieval filtering")
                
                if context_recall is not None and context_recall < 0.5:
                    report.append("  - *Recall concern*: Missing important information - increase retrieval scope or improve document coverage")
                
                # Metadata information
                metadata = ragas_results.get('metadata', {})
                eval_time = metadata.get('evaluation_time')
                if eval_time:
                    report.append(f"  - *Evaluation completed in {eval_time:.2f} seconds*")
                
            else:
                report.append("")
                report.append("## 🎯 RAGAS (RAG Assessment) Metrics")
                report.append("")
                if 'error' in ragas_results:
                    report.append(f"❌ **RAGAS evaluation failed**: {ragas_results['error']}")
                    report.append("  - *Installation*: Ensure RAGAS is installed with: `pip install ragas>=0.1.0`")
                    report.append("  - *Dependencies*: Check that all RAGAS dependencies are available")
                    report.append("  - *Configuration*: Verify RAGAS configuration in conf/evaluation/ragas_config.yaml")
                else:
                    report.append("❌ **RAGAS evaluation skipped**: RAGAS evaluator not available")
                    report.append("  - *Setup*: Install RAGAS package and enable in configuration")
                    report.append("  - *Benefits*: RAGAS provides specialized RAG-specific evaluation metrics")
                report.append("")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
    
    def run_full_evaluation(self, bleu_rouge_file: str = None, chatgpt_file: str = None) -> Dict[str, Any]:
        """Run complete evaluation suite.
        
        Args:
            bleu_rouge_file: Optional path to BLEU/ROUGE evaluation file.
            chatgpt_file: Optional path to ChatGPT comparison file.
            
        Returns:
            Complete evaluation results dictionary.
        """
        evaluation_results = {}
        
        # Default test questions for ChromaDB metrics
        test_questions = [
            "What is the Sample Educational Program?",
            "How long is the program?",
            "What are the fees for the program?",
            "What happens if I fail the technical assessment?",
            "Will I get a job after the program?"
        ]
        
        # 1. ChromaDB built-in metrics
        print("Running ChromaDB metrics evaluation...")
        chromadb_results = self.evaluate_chromadb_metrics(test_questions)
        evaluation_results['chromadb_metrics'] = chromadb_results
        
        # 2. BLEU/ROUGE evaluation
        if bleu_rouge_file and Path(bleu_rouge_file).exists():
            print("Running BLEU/ROUGE evaluation...")
            bleu_rouge_results = self.evaluate_bleu_rouge(bleu_rouge_file)
            evaluation_results['bleu_rouge'] = bleu_rouge_results
        else:
            print("BLEU/ROUGE evaluation skipped - file not found or not provided")
        
        # 3. ChatGPT comparison
        if chatgpt_file and Path(chatgpt_file).exists():
            print("Running ChatGPT comparison...")
            chatgpt_results = self.evaluate_chatgpt_comparison(chatgpt_file)
            evaluation_results['chatgpt_comparison'] = chatgpt_results
        else:
            print("ChatGPT comparison skipped - file not found or not provided")
        
        # 4. RAGAS evaluation
        if self.ragas_evaluator:
            print("Running RAGAS evaluation...")
            # Load ground truths if available from BLEU/ROUGE file
            ground_truths = None
            if bleu_rouge_file and Path(bleu_rouge_file).exists():
                try:
                    data_loader = EvaluationDataLoader(self.config)
                    with open(bleu_rouge_file, 'r', encoding='utf-8') as f:
                        bleu_data = json.load(f)
                    if isinstance(bleu_data, list) and bleu_data:
                        if 'reference_answer' in bleu_data[0]:
                            ground_truths = [item['reference_answer'] for item in bleu_data if 'reference_answer' in item]
                        elif 'ground_truth' in bleu_data[0]:
                            ground_truths = [item['ground_truth'] for item in bleu_data if 'ground_truth' in item]
                except Exception as e:
                    self.logger.warning(f"Could not load ground truths for RAGAS: {str(e)}")
            
            ragas_results = self.evaluate_ragas_metrics(test_questions, ground_truths)
            evaluation_results['ragas'] = ragas_results
        else:
            print("RAGAS evaluation skipped - RAGAS evaluator not available")
        
        return evaluation_results


def main() -> None:
    """Run evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG System')
    parser.add_argument('--bleu-rouge', help='Path to BLEU/ROUGE evaluation file')
    parser.add_argument('--chatgpt', help='Path to ChatGPT comparison file') 
    parser.add_argument('--output', default='evaluation_results.json', help='Output file for results')
    parser.add_argument('--markdown', help='Generate Markdown report (e.g., evaluation_report.md)')
    parser.add_argument('--config', help='RAG system config file path')
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(config_path=args.config)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_pipeline, args.config)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        bleu_rouge_file=args.bleu_rouge,
        chatgpt_file=args.chatgpt
    )
    
    # Save JSON results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Evaluation complete. Results saved to {args.output}")
    
    # Generate Markdown report if requested
    if args.markdown:
        evaluator.generate_markdown_report(results, args.markdown)
        print(f"Markdown report saved to {args.markdown}")
    
    # Print summary
    if 'chromadb_metrics' in results:
        chromadb = results['chromadb_metrics']['aggregate_metrics']
        print(f"\nChromaDB Metrics Summary:")
        print(f"  Success Rate: {chromadb['success_rate']:.2%}")
        print(f"  Avg Confidence: {chromadb['avg_retrieval_confidence']:.3f}")
        print(f"  Avg Sources: {chromadb['avg_num_sources']:.1f}")
    
    if 'bleu_rouge' in results and 'aggregate_metrics' in results['bleu_rouge']:
        bleu_rouge = results['bleu_rouge']['aggregate_metrics']
        if 'error' not in bleu_rouge and 'avg_bleu_score' in bleu_rouge:
            print(f"\nBLEU/ROUGE Summary:")
            print(f"  Avg BLEU: {bleu_rouge['avg_bleu_score']:.3f}")
            print(f"  Avg ROUGE-L: {bleu_rouge['avg_rougeL_f1']:.3f}")
            print(f"  Avg Semantic Similarity: {bleu_rouge['avg_semantic_similarity']:.3f}")
        else:
            print(f"\nBLEU/ROUGE Summary:")
            print(f"  ❌ Evaluation failed: {bleu_rouge.get('error', 'No successful evaluations')}")
    
    if 'chatgpt_comparison' in results and 'aggregate_metrics' in results['chatgpt_comparison']:
        chatgpt = results['chatgpt_comparison']['aggregate_metrics']
        if 'error' not in chatgpt and 'avg_response_similarity' in chatgpt:
            print(f"\nChatGPT Comparison Summary:")
            print(f"  Avg Response Similarity: {chatgpt['avg_response_similarity']:.3f}")
            print(f"  Avg Length Difference: {chatgpt['avg_length_difference']:.1f} words")
        else:
            print(f"\nChatGPT Comparison Summary:")
            print(f"  ❌ Comparison failed: {chatgpt.get('error', 'No successful comparisons')}")


if __name__ == "__main__":
    main()