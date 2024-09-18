# Singleton to properly handle the ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# Global variable to hold the singleton instance
_executor_instance = None

def get_executor() -> ThreadPoolExecutor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = ThreadPoolExecutor()
    return _executor_instance

def shutdown_executor():
    global _executor_instance
    if _executor_instance is not None:
        _executor_instance.shutdown(wait=True)
        _executor_instance = None
