"""
Performance monitoring and metrics collection utilities.

This module provides decorators and utilities for measuring performance,
collecting timing metrics, and benchmarking RAG pipeline operations.

Classes:
    PerformanceTracker: Context manager for timing operations
    MetricsCollector: Centralized metrics collection and reporting

Functions:
    time_operation: Decorator for timing function execution
    log_performance: Decorator for logging performance metrics
"""

import time
import logging
import functools
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Container for performance measurement data."""
    
    operation: str
    duration: float
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_type: Optional[str] = None


class MetricsCollector:
    """
    Centralized performance metrics collection and analysis.
    
    Collects timing data, operation success rates, and provides
    statistical analysis of system performance over time.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """
        Record a performance metric.
        
        Args:
            metric: Performance metric to record
        """
        self.metrics.append(metric)
        self.operation_stats[metric.operation].append(metric.duration)
        
        # Log significant performance issues
        if metric.duration > 10.0:  # More than 10 seconds
            logger.warning(f"Slow operation detected: {metric.operation} took {metric.duration:.2f}s")
        
        if not metric.success:
            logger.error(f"Failed operation: {metric.operation} - {metric.error_type}")
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """
        Get statistical summary for a specific operation.
        
        Args:
            operation: Name of the operation to analyze
            
        Returns:
            Dictionary containing statistical metrics
        """
        durations = list(self.operation_stats[operation])
        if not durations:
            return {"count": 0}
        
        return {
            "count": len(durations),
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "min": min(durations),
            "max": max(durations),
            "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0.0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all recorded operations."""
        return {op: self.get_operation_stats(op) for op in self.operation_stats.keys()}
    
    def get_success_rate(self, operation: str = None) -> float:
        """
        Calculate success rate for operations.
        
        Args:
            operation: Specific operation to analyze, or None for all operations
            
        Returns:
            Success rate as a percentage (0-100)
        """
        relevant_metrics = [m for m in self.metrics 
                           if operation is None or m.operation == operation]
        
        if not relevant_metrics:
            return 0.0
        
        successful = sum(1 for m in relevant_metrics if m.success)
        return (successful / len(relevant_metrics)) * 100
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to JSON file for analysis.
        
        Args:
            filepath: Path to save metrics JSON file
        """
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_metrics": len(self.metrics),
            "operation_stats": self.get_all_stats(),
            "overall_success_rate": self.get_success_rate(),
            "recent_metrics": [
                {
                    "operation": m.operation,
                    "duration": m.duration,
                    "timestamp": m.timestamp.isoformat(),
                    "success": m.success,
                    "parameters": m.parameters
                }
                for m in list(self.metrics)[-50:]  # Last 50 metrics
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _global_metrics


@contextmanager
def PerformanceTracker(operation: str, 
                      parameters: Dict[str, Any] = None,
                      metadata: Dict[str, Any] = None):
    """
    Context manager for tracking operation performance.
    
    Args:
        operation: Name of the operation being tracked
        parameters: Operation parameters for context
        metadata: Additional metadata about the operation
        
    Example:
        with PerformanceTracker("document_retrieval", {"k": 5, "query": "test"}):
            documents = retriever.retrieve_documents("test query", k=5)
    """
    start_time = time.time()
    metric = PerformanceMetric(
        operation=operation,
        duration=0.0,
        timestamp=datetime.now(),
        parameters=parameters or {},
        metadata=metadata or {}
    )
    
    try:
        yield metric
        metric.success = True
    except Exception as e:
        metric.success = False
        metric.error_type = type(e).__name__
        logger.error(f"Operation {operation} failed: {str(e)}")
        raise
    finally:
        metric.duration = time.time() - start_time
        _global_metrics.record_metric(metric)


def time_operation(operation_name: str = None):
    """
    Decorator for timing function execution and recording metrics.
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
        
    Example:
        @time_operation("document_embedding")
        def embed_documents(self, documents):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Extract parameters for logging (avoid large objects)
            safe_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    safe_kwargs[k] = v
                elif isinstance(v, (list, dict)) and len(str(v)) < 200:
                    safe_kwargs[k] = v
                else:
                    safe_kwargs[k] = f"<{type(v).__name__}>"
            
            with PerformanceTracker(op_name, parameters=safe_kwargs) as tracker:
                result = func(*args, **kwargs)
                
                # Add result metadata if reasonable size
                if isinstance(result, (dict, list)) and len(str(result)) < 500:
                    tracker.metadata['result_summary'] = str(result)[:200]
                elif hasattr(result, '__len__'):
                    tracker.metadata['result_length'] = len(result)
                
                return result
        
        return wrapper
    return decorator


def log_performance(level: str = "INFO"):
    """
    Decorator for logging performance metrics at specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = logging.getLogger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                log_msg = f"{func.__name__} completed in {duration:.3f}s"
                getattr(func_logger, level.lower())(log_msg)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_msg = f"{func.__name__} failed after {duration:.3f}s: {str(e)}"
                func_logger.error(log_msg)
                raise
        
        return wrapper
    return decorator


class BenchmarkSuite:
    """
    Benchmark suite for performance testing RAG pipeline components.
    
    Provides standardized performance tests that can be run to establish
    baselines and detect performance regressions.
    """
    
    def __init__(self, config_path: str = "conf"):
        """
        Initialize benchmark suite.
        
        Args:
            config_path: Path to system configuration
        """
        self.config_path = config_path
        self.results: List[Dict[str, Any]] = []
    
    def benchmark_retrieval(self, 
                           retriever, 
                           test_queries: List[str],
                           k_values: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark document retrieval performance.
        
        Args:
            retriever: DocumentRetriever instance to test
            test_queries: List of queries to test with
            k_values: List of k values to test (default: [1, 5, 10])
            
        Returns:
            Dictionary containing benchmark results
        """
        k_values = k_values or [1, 5, 10]
        results = {
            "operation": "document_retrieval",
            "test_queries": len(test_queries),
            "k_values": k_values,
            "results": {}
        }
        
        logger.info(f"Benchmarking retrieval with {len(test_queries)} queries")
        
        for k in k_values:
            k_results = []
            
            for query in test_queries:
                with PerformanceTracker(f"retrieve_k_{k}", {"query": query[:50], "k": k}) as tracker:
                    try:
                        documents = retriever.retrieve_documents(query, k=k)
                        k_results.append({
                            "query": query[:50],
                            "duration": tracker.duration,
                            "documents_returned": len(documents),
                            "success": True
                        })
                    except Exception as e:
                        k_results.append({
                            "query": query[:50], 
                            "duration": tracker.duration,
                            "documents_returned": 0,
                            "success": False,
                            "error": str(e)
                        })
            
            # Calculate statistics for this k value
            successful_results = [r for r in k_results if r["success"]]
            if successful_results:
                durations = [r["duration"] for r in successful_results]
                results["results"][f"k_{k}"] = {
                    "success_rate": len(successful_results) / len(k_results),
                    "avg_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations)
                }
            else:
                results["results"][f"k_{k}"] = {"success_rate": 0.0}
        
        self.results.append(results)
        return results
    
    def benchmark_generation(self, 
                           generator,
                           test_queries: List[str], 
                           sample_contexts: List[List[Document]]) -> Dict[str, Any]:
        """
        Benchmark answer generation performance.
        
        Args:
            generator: AnswerGenerator instance to test
            test_queries: List of test questions
            sample_contexts: List of document contexts for each query
            
        Returns:
            Dictionary containing benchmark results
        """
        results = {
            "operation": "answer_generation", 
            "test_queries": len(test_queries),
            "results": []
        }
        
        logger.info(f"Benchmarking generation with {len(test_queries)} queries")
        
        for i, (query, context) in enumerate(zip(test_queries, sample_contexts)):
            with PerformanceTracker(f"generate_answer", {"query": query[:50]}) as tracker:
                try:
                    response = generator.generate_answer(query, context)
                    results["results"].append({
                        "query_index": i,
                        "duration": tracker.duration,
                        "response_length": len(response.get("answer", "")),
                        "success": True
                    })
                except Exception as e:
                    results["results"].append({
                        "query_index": i,
                        "duration": tracker.duration, 
                        "success": False,
                        "error": str(e)
                    })
        
        # Calculate overall statistics
        successful_results = [r for r in results["results"] if r["success"]]
        if successful_results:
            durations = [r["duration"] for r in successful_results]
            results["summary"] = {
                "success_rate": len(successful_results) / len(results["results"]),
                "avg_duration": statistics.mean(durations),
                "median_duration": statistics.median(durations),
                "total_duration": sum(durations)
            }
        
        self.results.append(results)
        return results
    
    def export_benchmark_results(self, filepath: str) -> None:
        """
        Export benchmark results to JSON file.
        
        Args:
            filepath: Path to save benchmark results
        """
        export_data = {
            "benchmark_time": datetime.now().isoformat(),
            "config_path": self.config_path,
            "results": self.results,
            "system_metrics": _global_metrics.get_all_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported benchmark results to {filepath}")


def time_operation(operation_name: str = None, 
                  log_level: str = "INFO",
                  include_params: bool = True):
    """
    Decorator for timing operations and recording performance metrics.
    
    Args:
        operation_name: Custom name for the operation
        log_level: Logging level for performance logs
        include_params: Whether to include function parameters in metrics
        
    Example:
        @time_operation("document_retrieval", log_level="DEBUG")
        def retrieve_documents(self, query: str, k: int = 5):
            # Implementation
            return documents
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__qualname__}"
            
            # Prepare parameters for logging
            params = {}
            if include_params:
                # Get method parameters (skip self/cls)
                param_names = func.__code__.co_varnames[1:func.__code__.co_argcount]
                for i, name in enumerate(param_names):
                    if i < len(args):
                        params[name] = args[i]
                
                # Add keyword arguments
                for k, v in kwargs.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        params[k] = v
                    else:
                        params[k] = f"<{type(v).__name__}>"
            
            start_time = time.time()
            metric = PerformanceMetric(
                operation=op_name,
                duration=0.0,
                timestamp=datetime.now(),
                parameters=params
            )
            
            try:
                result = func(*args, **kwargs)
                metric.success = True
                metric.duration = time.time() - start_time
                
                # Log performance
                log_func = getattr(logger, log_level.lower())
                log_func(f"✅ {op_name} completed in {metric.duration:.3f}s")
                
                return result
                
            except Exception as e:
                metric.success = False
                metric.error_type = type(e).__name__
                metric.duration = time.time() - start_time
                
                logger.error(f"❌ {op_name} failed after {metric.duration:.3f}s: {str(e)}")
                raise
                
            finally:
                _global_metrics.record_metric(metric)
        
        return wrapper
    return decorator


@contextmanager 
def performance_context(operation: str, **context_params):
    """
    Context manager for performance tracking with additional context.
    
    Args:
        operation: Name of the operation
        **context_params: Additional context parameters to include
        
    Example:
        with performance_context("batch_processing", batch_size=100):
            process_documents(documents)
    """
    start_time = time.time()
    logger.debug(f"🚀 Starting {operation}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"✅ {operation} completed in {duration:.3f}s")
        
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=datetime.now(),
            parameters=context_params,
            success=True
        )
        _global_metrics.record_metric(metric)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ {operation} failed after {duration:.3f}s: {str(e)}")
        
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=datetime.now(),
            parameters=context_params,
            success=False,
            error_type=type(e).__name__
        )
        _global_metrics.record_metric(metric)
        raise


def create_performance_report() -> Dict[str, Any]:
    """
    Create a comprehensive performance report.
    
    Returns:
        Dictionary containing performance analysis and recommendations
    """
    metrics = _global_metrics
    all_stats = metrics.get_all_stats()
    
    # Identify slow operations
    slow_operations = []
    for op, stats in all_stats.items():
        if stats.get("mean", 0) > 5.0:  # Slower than 5 seconds on average
            slow_operations.append({
                "operation": op,
                "avg_duration": stats["mean"],
                "max_duration": stats["max"]
            })
    
    # Calculate overall system performance
    total_operations = sum(stats.get("count", 0) for stats in all_stats.values())
    overall_success_rate = metrics.get_success_rate()
    
    return {
        "report_time": datetime.now().isoformat(),
        "total_operations": total_operations,
        "overall_success_rate": overall_success_rate,
        "operation_statistics": all_stats,
        "slow_operations": slow_operations,
        "recommendations": _generate_performance_recommendations(all_stats, slow_operations)
    }


def _generate_performance_recommendations(stats: Dict[str, Dict], 
                                        slow_ops: List[Dict]) -> List[str]:
    """Generate performance improvement recommendations."""
    recommendations = []
    
    if slow_ops:
        recommendations.append(f"Consider optimizing {len(slow_ops)} slow operations")
    
    # Check for high variance operations
    high_variance_ops = []
    for op, op_stats in stats.items():
        if op_stats.get("count", 0) > 5:  # Enough data points
            mean_time = op_stats.get("mean", 0)
            std_dev = op_stats.get("std_dev", 0)
            if mean_time > 0 and (std_dev / mean_time) > 0.5:  # High coefficient of variation
                high_variance_ops.append(op)
    
    if high_variance_ops:
        recommendations.append(f"Investigate inconsistent performance in: {', '.join(high_variance_ops)}")
    
    # Check success rates
    low_success_ops = []
    for op in stats.keys():
        success_rate = _global_metrics.get_success_rate(op)
        if success_rate < 95.0:
            low_success_ops.append(f"{op} ({success_rate:.1f}%)")
    
    if low_success_ops:
        recommendations.append(f"Improve reliability for: {', '.join(low_success_ops)}")
    
    return recommendations