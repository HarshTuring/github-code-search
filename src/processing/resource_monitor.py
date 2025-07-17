import os
import psutil
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ResourceAwareThreadPool:
    """
    Thread pool that monitors system resources and adjusts worker count.
    """
    
    def __init__(self, 
                 min_workers=2, 
                 max_workers=None, 
                 memory_threshold=80,  # 80% memory usage threshold
                 cpu_threshold=90,     # 90% CPU usage threshold
                 check_interval=5):    # Check resources every 5 seconds
        """
        Initialize the resource-aware thread pool.
        
        Args:
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads (default: CPU count)
            memory_threshold: Memory usage percentage threshold to reduce workers
            cpu_threshold: CPU usage percentage threshold to reduce workers
            check_interval: How often to check resource usage (seconds)
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or os.cpu_count()
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.check_interval = check_interval
        
        # Start with a sensible default (half of max)
        self.current_workers = min(self.max_workers, max(self.min_workers, self.max_workers // 2))
        
        # Create the thread pool
        self.pool = ThreadPoolExecutor(max_workers=self.current_workers)
        self.pool_lock = threading.Lock()
        
        # Flag for stopping the monitor
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"ResourceAwareThreadPool initialized with {self.current_workers} workers "
                   f"(min={self.min_workers}, max={self.max_workers})")
    
    def _monitor_resources(self):
        """Monitor system resources and adjust thread pool size."""
        while self.running:
            try:
                # Get current resource usage
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Log current status
                logger.debug(f"System resources: Memory={memory_percent}%, CPU={cpu_percent}%, "
                            f"Current workers={self.current_workers}")
                
                # Calculate optimal worker count
                new_worker_count = self.current_workers
                
                # Decrease workers if resource usage is too high
                if memory_percent > self.memory_threshold or cpu_percent > self.cpu_threshold:
                    new_worker_count = max(self.min_workers, self.current_workers - 1)
                    logger.info(f"High resource usage detected: Memory={memory_percent}%, "
                               f"CPU={cpu_percent}%. Reducing workers to {new_worker_count}.")
                    
                # Increase workers if resource usage is low
                elif (memory_percent < self.memory_threshold * 0.7 and 
                      cpu_percent < self.cpu_threshold * 0.7 and
                      self.current_workers < self.max_workers):
                    new_worker_count = min(self.max_workers, self.current_workers + 1)
                    logger.info(f"Low resource usage detected: Memory={memory_percent}%, "
                               f"CPU={cpu_percent}%. Increasing workers to {new_worker_count}.")
                
                # Update thread pool if worker count changed
                if new_worker_count != self.current_workers:
                    with self.pool_lock:
                        # Create a new pool with the updated worker count
                        old_pool = self.pool
                        self.pool = ThreadPoolExecutor(max_workers=new_worker_count)
                        self.current_workers = new_worker_count
                        
                        # Shutdown the old pool without canceling running tasks
                        old_pool.shutdown(wait=False)
                
                # Sleep for the specified interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                time.sleep(self.check_interval)
    
    def submit(self, fn, *args, **kwargs):
        """Submit a task to the thread pool."""
        with self.pool_lock:
            return self.pool.submit(fn, *args, **kwargs)
    
    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Map a function to each item in the iterables."""
        with self.pool_lock:
            return self.pool.map(fn, *iterables, timeout=timeout, chunksize=chunksize)
    
    def shutdown(self, wait=True):
        """Shutdown the thread pool."""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        with self.pool_lock:
            self.pool.shutdown(wait=wait)
        
        logger.info("ResourceAwareThreadPool shut down")