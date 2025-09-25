"""
Production health monitoring and alerting system.

This module provides comprehensive health monitoring capabilities for the RAG pipeline,
including Prometheus metrics collection, health check endpoints, and alerting integration.

Classes:
    HealthMonitor: Central health monitoring coordinator
    PrometheusMetrics: Prometheus metrics collection and exposition
    AlertManager: Alert generation and notification system
"""

import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Info = None

from ..utils.performance import get_metrics_collector
from ..interfaces.base_interfaces import RetrievalError, GenerationError

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Container for component health status."""
    
    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class PrometheusMetrics:
    """
    Prometheus metrics collection for RAG pipeline monitoring.
    
    Provides standardized metrics collection for all pipeline components
    with proper labeling and metric types.
    """
    
    def __init__(self):
        """Initialize Prometheus metrics collectors."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - metrics disabled")
            return
        
        # Request metrics
        self.query_requests = Counter(
            'rag_query_requests_total',
            'Total number of RAG queries processed',
            ['status', 'component']
        )
        
        self.query_duration = Histogram(
            'rag_query_duration_seconds',
            'Time spent processing RAG queries',
            ['component', 'operation'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        # System metrics
        self.active_connections = Gauge(
            'rag_active_connections',
            'Number of active connections to RAG system'
        )
        
        self.vector_db_documents = Gauge(
            'rag_vector_db_documents_total',
            'Total number of documents in vector database'
        )
        
        # Error metrics
        self.error_count = Counter(
            'rag_errors_total',
            'Total number of errors by type and component',
            ['error_type', 'component']
        )
        
        # Component health
        self.component_health = Gauge(
            'rag_component_health_status',
            'Health status of RAG components (1=healthy, 0.5=degraded, 0=unhealthy)',
            ['component']
        )
        
        # System info
        self.system_info = Info(
            'rag_system_info',
            'RAG system information and configuration'
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_query(self, component: str, duration: float, status: str = "success"):
        """Record a query execution."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.query_requests.labels(status=status, component=component).inc()
        self.query_duration.labels(component=component, operation="query").observe(duration)
    
    def record_error(self, component: str, error_type: str):
        """Record an error occurrence."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.error_count.labels(error_type=error_type, component=component).inc()
    
    def update_component_health(self, component: str, status: str):
        """Update component health status."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        status_value = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}.get(status, 0.0)
        self.component_health.labels(component=component).set(status_value)
    
    def update_system_metrics(self, vector_db_docs: int, active_conns: int):
        """Update system-level metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.vector_db_documents.set(vector_db_docs)
        self.active_connections.set(active_conns)
    
    def set_system_info(self, version: str, config_info: Dict[str, str]):
        """Set system information metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        info_dict = {"version": version, **config_info}
        self.system_info.info(info_dict)


class AlertManager:
    """
    Alert generation and notification system for critical issues.
    
    Monitors system health and generates alerts when components
    enter degraded or unhealthy states.
    """
    
    def __init__(self, alert_thresholds: Dict[str, Dict[str, float]] = None):
        """
        Initialize alert manager.
        
        Args:
            alert_thresholds: Thresholds for different alert types
        """
        self.alert_thresholds = alert_thresholds or {
            "response_time": {"warning": 5.0, "critical": 15.0},
            "error_rate": {"warning": 0.05, "critical": 0.10},  # 5% and 10%
            "success_rate": {"warning": 0.95, "critical": 0.90}  # Below 95% and 90%
        }
        
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        logger.info("Alert manager initialized")
    
    def check_alerts(self, health_data: Dict[str, HealthStatus]) -> List[Dict[str, Any]]:
        """
        Check current system state and generate alerts if needed.
        
        Args:
            health_data: Current health status of all components
            
        Returns:
            List of active alerts
        """
        new_alerts = []
        
        for component, health in health_data.items():
            # Check response time alerts
            if health.response_time > self.alert_thresholds["response_time"]["critical"]:
                alert = self._create_alert(
                    "critical",
                    f"{component}_response_time",
                    f"Component {component} response time ({health.response_time:.2f}s) exceeds critical threshold"
                )
                new_alerts.append(alert)
            elif health.response_time > self.alert_thresholds["response_time"]["warning"]:
                alert = self._create_alert(
                    "warning",
                    f"{component}_response_time",
                    f"Component {component} response time ({health.response_time:.2f}s) exceeds warning threshold"
                )
                new_alerts.append(alert)
            
            # Check component status alerts
            if health.status == "unhealthy":
                alert = self._create_alert(
                    "critical",
                    f"{component}_unhealthy",
                    f"Component {component} is unhealthy: {health.error_message}"
                )
                new_alerts.append(alert)
            elif health.status == "degraded":
                alert = self._create_alert(
                    "warning",
                    f"{component}_degraded",
                    f"Component {component} is degraded"
                )
                new_alerts.append(alert)
        
        # Update active alerts
        for alert in new_alerts:
            self.active_alerts[alert["alert_id"]] = alert
            self.alert_history.append(alert)
        
        return list(self.active_alerts.values())
    
    def _create_alert(self, severity: str, alert_type: str, message: str) -> Dict[str, Any]:
        """Create an alert record."""
        return {
            "alert_id": f"{alert_type}_{int(time.time())}",
            "severity": severity,
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "resolved": False
        }
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["resolved"] = True
            self.active_alerts[alert_id]["resolved_at"] = datetime.now().isoformat()
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")
            return True
        return False


class HealthMonitor:
    """
    Central health monitoring coordinator for RAG pipeline.
    
    Orchestrates health checks across all components, collects metrics,
    and provides comprehensive system health reporting.
    """
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 enable_prometheus: bool = True,
                 prometheus_port: int = 9090):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Interval between health checks in seconds
            enable_prometheus: Whether to enable Prometheus metrics
            prometheus_port: Port for Prometheus metrics server
        """
        self.check_interval = check_interval
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port
        
        # Component health tracking
        self.component_health: Dict[str, HealthStatus] = {}
        self.health_check_functions: Dict[str, Callable] = {}
        
        # Initialize subsystems
        if self.enable_prometheus:
            self.prometheus = PrometheusMetrics()
            self._start_prometheus_server()
        else:
            self.prometheus = None
            logger.warning("Prometheus metrics disabled")
        
        self.alert_manager = AlertManager()
        self.metrics_collector = get_metrics_collector()
        
        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info(f"Health monitor initialized with {check_interval}s check interval")
    
    def register_component(self, component_name: str, health_check_func: Callable) -> None:
        """
        Register a component for health monitoring.
        
        Args:
            component_name: Name of the component to monitor
            health_check_func: Function that returns health status dict
        """
        self.health_check_functions[component_name] = health_check_func
        logger.info(f"Registered component for monitoring: {component_name}")
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Background health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Background health monitoring stopped")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            Dictionary containing overall health status and component details
        """
        # Perform health checks for all registered components
        component_statuses = {}
        for component_name, health_func in self.health_check_functions.items():
            try:
                start_time = time.time()
                health_result = health_func()
                response_time = time.time() - start_time
                
                status = HealthStatus(
                    component=component_name,
                    status=health_result.get("status", "unknown"),
                    last_check=datetime.now(),
                    response_time=response_time,
                    error_message=health_result.get("error"),
                    metadata=health_result.get("diagnostics", {})
                )
                
                component_statuses[component_name] = status
                self.component_health[component_name] = status
                
                # Update Prometheus metrics
                if self.prometheus:
                    self.prometheus.update_component_health(component_name, status.status)
                
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {str(e)}")
                status = HealthStatus(
                    component=component_name,
                    status="unhealthy",
                    last_check=datetime.now(),
                    response_time=0.0,
                    error_message=str(e)
                )
                component_statuses[component_name] = status
        
        # Calculate overall system health
        overall_status = self._calculate_overall_health(component_statuses)
        
        # Check for alerts
        active_alerts = self.alert_manager.check_alerts(component_statuses)
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {name: {
                "status": status.status,
                "response_time": status.response_time,
                "last_check": status.last_check.isoformat(),
                "error_message": status.error_message,
                "metadata": status.metadata
            } for name, status in component_statuses.items()},
            "active_alerts": active_alerts,
            "system_metrics": system_metrics,
            "performance_summary": self._get_performance_summary()
        }
    
    def _calculate_overall_health(self, component_statuses: Dict[str, HealthStatus]) -> str:
        """Calculate overall system health from component statuses."""
        if not component_statuses:
            return "unknown"
        
        statuses = [status.status for status in component_statuses.values()]
        
        if any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        elif any(status == "degraded" for status in statuses):
            return "degraded"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics (CPU, memory, disk)."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Update Prometheus system metrics
            if self.prometheus:
                # Add system metrics to Prometheus if needed
                pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return {"error": str(e)}
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from metrics collector."""
        try:
            all_stats = self.metrics_collector.get_all_stats()
            
            summary = {
                "total_operations": sum(stats.get("count", 0) for stats in all_stats.values()),
                "avg_response_times": {
                    op: stats.get("mean", 0) 
                    for op, stats in all_stats.items() 
                    if stats.get("count", 0) > 0
                },
                "slowest_operations": sorted([
                    {"operation": op, "avg_time": stats.get("mean", 0)}
                    for op, stats in all_stats.items()
                    if stats.get("count", 0) > 0
                ], key=lambda x: x["avg_time"], reverse=True)[:5]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {str(e)}")
            return {"error": str(e)}
    
    def _start_prometheus_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {str(e)}")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("Starting health monitoring loop")
        
        while self._monitoring_active:
            try:
                start_time = time.time()
                
                # Perform health checks
                health_status = self.get_system_health()
                
                # Log health summary
                overall_status = health_status["overall_status"]
                component_count = len(health_status["components"])
                active_alerts = len(health_status["active_alerts"])
                
                logger.info(f"Health check completed: {overall_status} "
                          f"({component_count} components, {active_alerts} alerts)")
                
                # Log any critical alerts
                for alert in health_status["active_alerts"]:
                    if alert["severity"] == "critical":
                        logger.critical(f"CRITICAL ALERT: {alert['message']}")
                
                # Wait for next check interval
                check_duration = time.time() - start_time
                sleep_time = max(0, self.check_interval - check_duration)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.check_interval)  # Continue monitoring despite errors


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor instance."""
    global _global_health_monitor
    
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    
    return _global_health_monitor


def setup_monitoring(retriever, generator, pipeline = None) -> HealthMonitor:
    """
    Setup monitoring for all RAG pipeline components.
    
    Args:
        retriever: DocumentRetriever instance
        generator: AnswerGenerator instance  
        pipeline: Optional RAGPipeline instance
        
    Returns:
        Configured HealthMonitor instance
    """
    monitor = get_health_monitor()
    
    # Register component health checks
    monitor.register_component("retriever", retriever.health_check)
    monitor.register_component("generator", generator.health_check)
    
    if pipeline:
        monitor.register_component("pipeline", lambda: {"status": "healthy", "component": "pipeline"})
    
    # Set system info in Prometheus
    if monitor.prometheus:
        monitor.prometheus.set_system_info("1.0.0", {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "embedding_model": getattr(retriever, 'embedding_model', 'unknown'),
            "llm_provider": getattr(generator, 'provider', 'unknown')
        })
    
    # Start monitoring
    monitor.start_monitoring()
    
    logger.info("RAG pipeline monitoring setup completed")
    return monitor


def create_health_endpoint(health_monitor: HealthMonitor) -> Callable:
    """
    Create a health check endpoint function for web frameworks.
    
    Args:
        health_monitor: HealthMonitor instance
        
    Returns:
        Function that returns health status suitable for HTTP endpoints
    """
    def health_endpoint():
        """HTTP endpoint for system health checks."""
        try:
            health_data = health_monitor.get_system_health()
            
            # Determine HTTP status code based on health
            status_code = 200
            if health_data["overall_status"] == "degraded":
                status_code = 200  # Still serving traffic
            elif health_data["overall_status"] == "unhealthy":
                status_code = 503  # Service unavailable
            
            return {
                "status_code": status_code,
                "response": health_data
            }
            
        except Exception as e:
            logger.error(f"Health endpoint error: {str(e)}")
            return {
                "status_code": 500,
                "response": {
                    "overall_status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    return health_endpoint


def shutdown_monitoring() -> None:
    """Shutdown global health monitoring."""
    global _global_health_monitor
    
    if _global_health_monitor:
        _global_health_monitor.stop_monitoring()
        _global_health_monitor = None
        logger.info("Health monitoring shutdown completed")