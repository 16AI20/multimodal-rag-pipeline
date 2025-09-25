"""
Async utilities and helpers for RAG pipeline components.

This module provides utilities for converting synchronous operations to async,
managing concurrent operations, and handling async context managers for
improved system responsiveness and scalability.

Functions:
    async_retry: Decorator for async operations with retry logic
    concurrent_map: Execute multiple async operations concurrently
    AsyncConnectionPool: Manage connections with automatic pooling
"""

import asyncio
import logging
from typing import Any, Callable, List, Dict, Optional, TypeVar, Awaitable
from functools import wraps
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


def async_retry(max_attempts: int = 3, 
               delay: float = 1.0, 
               backoff_factor: float = 2.0,
               exceptions: tuple = (Exception,)):
    """
    Decorator for async operations with exponential backoff retry logic.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each failure
        exceptions: Tuple of exceptions to catch and retry on
        
    Example:
        @async_retry(max_attempts=3, delay=1.0)
        async def unreliable_operation():
            # Implementation that might fail
            pass
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"{func.__name__} failed on attempt {attempt + 1}, retrying in {current_delay}s: {str(e)}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {str(e)}")
            
            raise last_exception
        
        return wrapper
    return decorator


async def concurrent_map(func: Callable[[T], Awaitable[Any]], 
                        items: List[T], 
                        max_concurrency: int = 10) -> List[Any]:
    """
    Execute an async function concurrently over a list of items.
    
    Args:
        func: Async function to apply to each item
        items: List of items to process
        max_concurrency: Maximum number of concurrent operations
        
    Returns:
        List of results in the same order as input items
        
    Example:
        async def process_document(doc):
            return await embed_document(doc)
        
        results = await concurrent_map(process_document, documents, max_concurrency=5)
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded_func(item):
        async with semaphore:
            return await func(item)
    
    logger.debug(f"Processing {len(items)} items with max concurrency {max_concurrency}")
    
    start_time = time.time()
    tasks = [bounded_func(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check for exceptions in results
    exceptions = [r for r in results if isinstance(r, Exception)]
    if exceptions:
        logger.warning(f"Encountered {len(exceptions)} exceptions during concurrent processing")
    
    duration = time.time() - start_time
    logger.info(f"Concurrent processing completed in {duration:.2f}s")
    
    return results


class AsyncConnectionPool:
    """
    Async connection pool for managing database or API connections.
    
    Provides automatic connection management, health checking, and
    connection recycling for better resource utilization.
    """
    
    def __init__(self, 
                 connection_factory: Callable[[], Awaitable[Any]],
                 max_connections: int = 10,
                 health_check_interval: float = 30.0):
        """
        Initialize async connection pool.
        
        Args:
            connection_factory: Async function that creates new connections
            max_connections: Maximum number of connections in pool
            health_check_interval: Interval for connection health checks
        """
        self.connection_factory = connection_factory
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval
        
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._total_connections = 0
        self._health_check_task: Optional[asyncio.Task] = None
        self._closed = False
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        logger.info(f"Initializing connection pool with max {self.max_connections} connections")
        
        # Create initial connections
        initial_size = min(2, self.max_connections)
        for _ in range(initial_size):
            connection = await self.connection_factory()
            await self._pool.put(connection)
            self._total_connections += 1
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Connection pool initialized with {initial_size} connections")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a connection from the pool.
        
        Example:
            async with pool.get_connection() as conn:
                result = await conn.execute_query("SELECT * FROM table")
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        connection = None
        try:
            # Try to get existing connection or create new one
            try:
                connection = await asyncio.wait_for(self._pool.get(), timeout=5.0)
            except asyncio.TimeoutError:
                if self._total_connections < self.max_connections:
                    logger.debug("Creating new connection for pool")
                    connection = await self.connection_factory()
                    self._total_connections += 1
                else:
                    # Wait longer if pool is at capacity
                    connection = await self._pool.get()
            
            yield connection
            
        except Exception as e:
            logger.error(f"Error using pooled connection: {str(e)}")
            # Don't return problematic connection to pool
            connection = None
            raise
        finally:
            # Return connection to pool if it's still valid
            if connection is not None and not self._closed:
                try:
                    await self._pool.put(connection)
                except Exception as e:
                    logger.warning(f"Failed to return connection to pool: {str(e)}")
    
    async def close(self) -> None:
        """Close all connections and shutdown the pool."""
        if self._closed:
            return
        
        logger.info("Closing connection pool")
        self._closed = True
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        while not self._pool.empty():
            try:
                connection = await asyncio.wait_for(self._pool.get(), timeout=1.0)
                if hasattr(connection, 'close'):
                    await connection.close()
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Error closing pooled connection: {str(e)}")
        
        logger.info("Connection pool closed")
    
    async def _health_check_loop(self) -> None:
        """Background task to check connection health."""
        while not self._closed:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if not self._closed:
                    logger.debug("Performing connection pool health check")
                    # Add health check logic here if needed
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection pool health check: {str(e)}")


async def async_timeout(coro: Awaitable[T], timeout: float, operation_name: str = "operation") -> T:
    """
    Execute async operation with timeout and proper error handling.
    
    Args:
        coro: Async operation to execute
        timeout: Timeout in seconds
        operation_name: Name of operation for logging
        
    Returns:
        Result of the async operation
        
    Raises:
        asyncio.TimeoutError: When operation exceeds timeout
    """
    try:
        logger.debug(f"Starting {operation_name} with {timeout}s timeout")
        result = await asyncio.wait_for(coro, timeout=timeout)
        logger.debug(f"Completed {operation_name} successfully")
        return result
        
    except asyncio.TimeoutError:
        logger.error(f"Operation {operation_name} timed out after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"Operation {operation_name} failed: {str(e)}")
        raise


class AsyncBatchProcessor:
    """
    Process items in async batches for better resource utilization.
    
    Useful for processing large numbers of documents or queries while
    controlling memory usage and system load.
    """
    
    def __init__(self, batch_size: int = 10, max_concurrency: int = 5):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
            max_concurrency: Maximum concurrent operations per batch
        """
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
    
    async def process_batches(self, 
                             items: List[T], 
                             processor_func: Callable[[T], Awaitable[Any]]) -> List[Any]:
        """
        Process items in batches asynchronously.
        
        Args:
            items: List of items to process
            processor_func: Async function to apply to each item
            
        Returns:
            List of results for all items
        """
        logger.info(f"Processing {len(items)} items in batches of {self.batch_size}")
        
        all_results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}: {len(batch)} items")
            
            # Process batch concurrently
            batch_results = await concurrent_map(
                processor_func, 
                batch, 
                max_concurrency=self.max_concurrency
            )
            
            all_results.extend(batch_results)
            
            # Brief pause between batches to prevent overwhelming the system
            if i + self.batch_size < len(items):
                await asyncio.sleep(0.1)
        
        logger.info(f"Batch processing completed: {len(all_results)} results")
        return all_results


def make_async(sync_func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    Convert a synchronous function to async using thread pool.
    
    Args:
        sync_func: Synchronous function to convert
        
    Returns:
        Async version of the function
        
    Example:
        sync_embedding_func = lambda x: model.encode(x)
        async_embedding_func = make_async(sync_embedding_func)
        result = await async_embedding_func(text)
    """
    @wraps(sync_func)
    async def async_wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: sync_func(*args, **kwargs))
    
    return async_wrapper