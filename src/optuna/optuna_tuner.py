"""
Optuna-based hyperparameter tuning for RAG pipeline.
Integrates with the evaluation pipeline to optimize retrieval and generation parameters.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Optional Optuna import - gracefully handle if not installed
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not installed. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False


class RAGHyperparameterTuner:
    """Hyperparameter tuner for RAG pipeline using Optuna."""
    
    def __init__(self, config: DictConfig, evaluator=None) -> None:
        """Initialize the RAG hyperparameter tuner.
        
        Args:
            config: Hydra configuration containing Optuna settings.
            evaluator: Optional evaluator instance for scoring trials.
        """
        self.config = config
        self.hp_config = config.optuna
        self.enabled = self.hp_config.get('enabled', False)
        self.evaluator = evaluator
        
        if self.enabled and not OPTUNA_AVAILABLE:
            logger.warning("Hyperparameter tuning enabled but Optuna not installed")
            self.enabled = False
        
        # Initialize study
        self.study = None
        if self.enabled:
            self._initialize_study()
        
        logger.info(f"Hyperparameter tuning: {'enabled' if self.enabled else 'disabled'}")
    
    def _initialize_study(self) -> None:
        """Initialize Optuna study with configuration.
        
        Creates an Optuna study with configured sampler, pruner, and storage.
        """
        try:
            study_config = self.hp_config.study
            
            # Set up sampler
            sampler = TPESampler(seed=42)
            
            # Set up pruner if enabled
            pruner = None
            if self.hp_config.trials.pruning.get('enabled', True):
                pruner = MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=self.hp_config.trials.pruning.get('patience', 10)
                )
            
            # Create or load study
            self.study = optuna.create_study(
                study_name=study_config.get('study_name', 'rag_optimization'),
                direction=study_config.get('direction', 'maximize'),
                storage=study_config.get('storage', 'sqlite:///hyperparameter_studies.db'),
                load_if_exists=study_config.get('load_if_exists', True),
                sampler=sampler,
                pruner=pruner
            )
            
            logger.info(f"Initialized Optuna study: {self.study.study_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Optuna study: {e}")
            self.enabled = False
    
    def suggest_parameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object for parameter suggestion.
            
        Returns:
            Dictionary of suggested hyperparameters.
        """
        params = {}
        param_config = self.hp_config.parameters
        
        # Retrieval parameters
        if 'retrieval' in param_config:
            retrieval_params = param_config.retrieval
            
            # Number of documents to retrieve
            if 'k' in retrieval_params:
                k_config = retrieval_params.k
                params['k'] = trial.suggest_int(
                    'k', k_config.low, k_config.high, step=k_config.get('step', 1)
                )
            
            # Chunk sizes
            for chunk_type in ['chunk_size_pdf', 'chunk_size_html']:
                if chunk_type in retrieval_params:
                    chunk_config = retrieval_params[chunk_type]
                    params[chunk_type] = trial.suggest_int(
                        chunk_type, chunk_config.low, chunk_config.high, 
                        step=chunk_config.get('step', 100)
                    )
            
            # Chunk overlap
            if 'chunk_overlap' in retrieval_params:
                overlap_config = retrieval_params.chunk_overlap
                params['chunk_overlap'] = trial.suggest_int(
                    'chunk_overlap', overlap_config.low, overlap_config.high,
                    step=overlap_config.get('step', 25)
                )
            
            # Similarity threshold
            if 'similarity_threshold' in retrieval_params:
                sim_config = retrieval_params.similarity_threshold
                params['similarity_threshold'] = trial.suggest_float(
                    'similarity_threshold', sim_config.low, sim_config.high,
                    step=sim_config.get('step', 0.1)
                )
            
            # Reranking
            if 'reranking_enabled' in retrieval_params:
                rerank_config = retrieval_params.reranking_enabled
                params['reranking_enabled'] = trial.suggest_categorical(
                    'reranking_enabled', rerank_config.choices
                )
                
                # Conditional reranking parameters
                if params.get('reranking_enabled') and 'reranking_top_k' in retrieval_params:
                    rerank_k_config = retrieval_params.reranking_top_k
                    params['reranking_top_k'] = trial.suggest_int(
                        'reranking_top_k', rerank_k_config.low, rerank_k_config.high,
                        step=rerank_k_config.get('step', 5)
                    )
        
        # Generation parameters
        if 'generation' in param_config:
            generation_params = param_config.generation
            
            # Temperature
            if 'temperature' in generation_params:
                temp_config = generation_params.temperature
                params['temperature'] = trial.suggest_float(
                    'temperature', temp_config.low, temp_config.high,
                    step=temp_config.get('step', 0.1)
                )
            
            # Max tokens
            if 'max_tokens' in generation_params:
                tokens_config = generation_params.max_tokens
                params['max_tokens'] = trial.suggest_int(
                    'max_tokens', tokens_config.low, tokens_config.high,
                    step=tokens_config.get('step', 64)
                )
            
            # Top-p
            if 'top_p' in generation_params:
                top_p_config = generation_params.top_p
                params['top_p'] = trial.suggest_float(
                    'top_p', top_p_config.low, top_p_config.high,
                    step=top_p_config.get('step', 0.05)
                )
        
        return params
    
    def create_trial_config(self, trial_params: Dict[str, Any]) -> DictConfig:
        """Create a temporary config with trial parameters.
        
        Args:
            trial_params: Dictionary of trial parameters to apply.
            
        Returns:
            Hydra config with trial parameters applied.
        """
        # Start with a deep copy of the base config
        trial_config = OmegaConf.create(OmegaConf.to_yaml(self.config))
        
        # Ensure all required sections exist
        if 'document_processing' not in trial_config:
            trial_config.document_processing = {}
        if 'chunk_sizes' not in trial_config.document_processing:
            trial_config.document_processing.chunk_sizes = {}
        if 'embeddings' not in trial_config:
            trial_config.embeddings = {}
        if 'llm' not in trial_config:
            trial_config.llm = {}
        
        # Apply trial parameters
        if 'k' in trial_params:
            # Store k for retrieval pipeline
            trial_config.retrieval_k = trial_params['k']
        
        if 'chunk_size_pdf' in trial_params:
            trial_config.document_processing.chunk_sizes.pdf = trial_params['chunk_size_pdf']
        
        if 'chunk_size_html' in trial_params:
            trial_config.document_processing.chunk_sizes.html = trial_params['chunk_size_html']
        
        if 'chunk_overlap' in trial_params:
            trial_config.document_processing.chunk_overlap = trial_params['chunk_overlap']
            
        if 'reranking_enabled' in trial_params:
            trial_config.embeddings.reranking_enabled = trial_params['reranking_enabled']
            
        if 'reranking_top_k' in trial_params:
            trial_config.embeddings.reranking_top_k = trial_params['reranking_top_k']
        
        if 'temperature' in trial_params:
            trial_config.llm.temperature = trial_params['temperature']
            
        if 'max_tokens' in trial_params:
            trial_config.llm.max_tokens = trial_params['max_tokens']
            
        if 'top_p' in trial_params:
            trial_config.llm.top_p = trial_params['top_p']
        
        return trial_config
    
    def objective_function(self, trial) -> float:
        """Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object.
            
        Returns:
            Optimization score to maximize.
        """
        try:
            # Get trial parameters
            trial_params = self.suggest_parameters(trial)
            
            # Create config for this trial
            trial_config = self.create_trial_config(trial_params)
            
            # Create temporary RAG pipeline with trial config
            from ..rag.pipeline import RAGPipeline
            
            # Initialize RAG pipeline with original config directory but override specific parameters
            # This avoids config loading issues while still applying trial parameters
            rag_pipeline = RAGPipeline(config_path="conf")
            
            # Override pipeline parameters with trial values
            if hasattr(rag_pipeline, 'retriever') and rag_pipeline.retriever:
                # Apply retrieval parameters
                if 'k' in trial_params:
                    rag_pipeline.retriever.default_k = trial_params['k']
                if 'reranking_enabled' in trial_params:
                    rag_pipeline.retriever.reranking_enabled = trial_params['reranking_enabled']
                if 'reranking_top_k' in trial_params and hasattr(rag_pipeline.retriever, 'reranking_top_k'):
                    rag_pipeline.retriever.reranking_top_k = trial_params['reranking_top_k']
            
            if hasattr(rag_pipeline, 'generator') and rag_pipeline.generator:
                # Apply generation parameters
                if 'temperature' in trial_params:
                    rag_pipeline.generator.temperature = trial_params['temperature']
                if 'max_tokens' in trial_params:
                    rag_pipeline.generator.max_length = trial_params['max_tokens']  # Note: uses max_length internally
                # Note: top_p not currently supported in generator, skip for now
            
            # Run evaluation
            eval_config = self.hp_config.evaluation
            
            results = {}
            total_score = 0.0
            weight_sum = 0.0
            
            # BLEU/ROUGE evaluation
            if eval_config.methods.get('bleu_rouge', True):
                from ..evaluation.rag_evaluator import RAGEvaluator
                evaluator = RAGEvaluator(rag_pipeline)
                
                bleu_rouge_results = evaluator.evaluate_bleu_rouge(eval_config.bleu_rouge_file)
                
                if 'aggregate_metrics' in bleu_rouge_results:
                    metrics = bleu_rouge_results['aggregate_metrics']
                    
                    # Add weighted scores
                    weights = eval_config.objective_weights
                    
                    if 'rouge_l_f1' in weights:
                        score = metrics.get('avg_rougeL_f1', 0.0) * weights['rouge_l_f1']
                        total_score += score
                        weight_sum += weights['rouge_l_f1']
                    
                    if 'bleu_score' in weights:
                        score = metrics.get('avg_bleu_score', 0.0) * weights['bleu_score']
                        total_score += score
                        weight_sum += weights['bleu_score']
                    
                    if 'semantic_similarity' in weights:
                        score = metrics.get('avg_semantic_similarity', 0.0) * weights['semantic_similarity']
                        total_score += score
                        weight_sum += weights['semantic_similarity']
            
            # Response time penalty
            if eval_config.methods.get('response_time', True):
                # Measure average response time using configurable test questions
                eval_config = self.config.get('evaluation', {})
                test_questions = eval_config.get('performance_test_questions', 
                                               ["What is the Sample Program?", "How long is the program?", "What are the fees?"])
                response_times = []
                
                for question in test_questions:
                    start_time = time.time()
                    try:
                        result = rag_pipeline.query(question, k=trial_params.get('k', 5))
                        end_time = time.time()
                        
                        if 'error' not in result:
                            response_times.append(end_time - start_time)
                        else:
                            logger.warning(f"Query failed during trial: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.warning(f"Query exception during trial: {e}")
                        # Continue with other test questions
                
                if response_times:
                    avg_response_time = np.mean(response_times)
                    
                    # Convert response time to penalty (faster = better)
                    max_time = eval_config.constraints.get('max_response_time', 30.0)
                    time_penalty = max(0, 1.0 - (avg_response_time / max_time))
                    
                    penalty_weight = eval_config.objective_weights.get('response_time_penalty', 0.2)
                    total_score += time_penalty * penalty_weight
                    weight_sum += penalty_weight
            
            # Normalize score
            if weight_sum > 0:
                final_score = total_score / weight_sum
            else:
                final_score = 0.0
            
            
            # Log trial results
            logger.info(f"Trial {trial.number}: Score = {final_score:.4f}, Params = {trial_params}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return 0.0  # Return worst possible score on error
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Returns:
            Dictionary containing optimization results and best parameters.
        """
        if not self.enabled:
            return {"error": "Hyperparameter tuning not enabled"}
        
        try:
            trials_config = self.hp_config.trials
            
            logger.info(f"Starting optimization with {trials_config.n_trials} trials")
            
            # Run optimization
            self.study.optimize(
                self.objective_function,
                n_trials=trials_config.get('n_trials', 50),
                timeout=trials_config.get('timeout', 3600),
                n_jobs=trials_config.get('n_jobs', 1)
            )
            
            # Get best trial
            best_trial = self.study.best_trial
            
            results = {
                "best_score": best_trial.value,
                "best_params": best_trial.params,
                "n_trials": len(self.study.trials),
                "study_name": self.study.study_name
            }
            
            # Save results
            self._save_results(results)
            
            # Export best config if requested
            if self.hp_config.results.get('export_best_config', True):
                self._export_best_config(best_trial.params)
            
            logger.info(f"Optimization completed. Best score: {best_trial.value:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return {"error": str(e)}
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results.
        
        Args:
            results: Optimization results dictionary to save.
        """
        try:
            output_dir = Path(self.hp_config.results.get('output_dir', 'hyperparameter_results'))
            output_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"optimization_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _export_best_config(self, best_params: Dict[str, Any]) -> None:
        """Export the best configuration as a new Hydra config.
        
        Args:
            best_params: Best parameters from optimization.
        """
        try:
            # Create optimized config
            best_config = self.create_trial_config(best_params)
            
            # Save to specified path
            output_path = self.hp_config.results.get('config_output_path', 'conf/tuned_config.yaml')
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                OmegaConf.save(best_config, f)
            
            logger.info(f"Best config exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting best config: {e}")
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history and statistics.
        
        Returns:
            Dictionary containing optimization history and metrics.
        """
        if not self.enabled or not self.study:
            return {"error": "No study available"}
        
        trials = self.study.trials
        
        return {
            "n_trials": len(trials),
            "best_score": self.study.best_value,
            "best_params": self.study.best_params,
            "trial_scores": [t.value for t in trials if t.value is not None],
            "study_name": self.study.study_name
        }