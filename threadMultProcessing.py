# Complete Guide to Python Threading and Multiprocessing
# =======================================================

"""
THREADING vs MULTIPROCESSING: Key Differences
=============================================

THREADING:
- Runs in the same process, shares memory
- Limited by GIL (Global Interpreter Lock) for CPU-intensive tasks
- Best for I/O-bound tasks (file operations, network requests, database queries)
- Lower memory overhead
- Shared state between threads (can cause race conditions)

MULTIPROCESSING:
- Runs in separate processes, isolated memory
- No GIL limitation, true parallelism for CPU-intensive tasks
- Best for CPU-bound tasks (calculations, data processing)
- Higher memory overhead
- Inter-process communication needed for data sharing
"""

# =============================================================================
# PART 1: THREADING
# =============================================================================

import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import queue

print("=" * 60)
print("THREADING EXAMPLES")
print("=" * 60)


# Example 1: Basic Threading
# --------------------------
def worker_task(name, delay):
    """Simple worker function"""
    print(f"Thread {name} starting")
    time.sleep(delay)
    print(f"Thread {name} finished after {delay} seconds")


# Creating and starting threads manually
print("\n1. Basic Threading:")
thread1 = threading.Thread(target=worker_task, args=("A", 2))
thread2 = threading.Thread(target=worker_task, args=("B", 1))

start_time = time.time()
thread1.start()
thread2.start()

# Wait for threads to complete
thread1.join()
thread2.join()
print(f"All threads completed in {time.time() - start_time:.2f} seconds")

# Example 2: Thread-Safe Operations with Lock
# -------------------------------------------
print("\n2. Thread-Safe Operations:")

counter = 0
lock = threading.Lock()


def increment_counter(name, iterations):
    """Thread-safe counter increment"""
    global counter
    for i in range(iterations):
        with lock:  # Acquire lock before modifying shared resource
            temp = counter
            time.sleep(0.0001)  # Simulate processing time
            counter = temp + 1
    print(f"Thread {name} completed {iterations} increments")


# Without lock, this could lead to race conditions
threads = []
for i in range(3):
    t = threading.Thread(target=increment_counter, args=(f"T{i}", 100))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final counter value: {counter} (should be 300)")

# Example 3: Producer-Consumer Pattern with Queue
# -----------------------------------------------
print("\n3. Producer-Consumer with Queue:")

task_queue = queue.Queue()
result_queue = queue.Queue()


def producer(num_tasks):
    """Produces tasks and puts them in queue"""
    for i in range(num_tasks):
        task = f"Task-{i}"
        task_queue.put(task)
        print(f"Produced: {task}")
        time.sleep(0.1)

    # Signal that we're done producing
    task_queue.put(None)


def consumer(name):
    """Consumes tasks from queue"""
    while True:
        task = task_queue.get()
        if task is None:
            # Poison pill - we're done
            task_queue.task_done()
            break

        # Process the task
        result = f"{task} processed by {name}"
        result_queue.put(result)
        print(f"Consumer {name}: {result}")
        time.sleep(0.2)
        task_queue.task_done()


# Start producer and consumers
producer_thread = threading.Thread(target=producer, args=(5,))
consumer1 = threading.Thread(target=consumer, args=("C1",))
consumer2 = threading.Thread(target=consumer, args=("C2",))

producer_thread.start()
consumer1.start()
consumer2.start()

producer_thread.join()
consumer1.join()
consumer2.join()

# Example 4: ThreadPoolExecutor (Recommended Approach)
# ---------------------------------------------------
print("\n4. ThreadPoolExecutor:")


def fetch_url(url):
    """Simulate fetching URL (I/O-bound task)"""
    try:
        # In real scenario, you'd use requests.get(url)
        time.sleep(1)  # Simulate network delay
        return f"Fetched {url} - Status: 200"
    except Exception as e:
        return f"Error fetching {url}: {e}"


urls = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
]

start_time = time.time()

# Using ThreadPoolExecutor for I/O-bound tasks
with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit all tasks
    future_to_url = {executor.submit(fetch_url, url): url for url in urls}

    # Collect results as they complete
    results = []
    for future in future_to_url:
        result = future.result()
        results.append(result)
        print(result)

print(f"All URL fetches completed in {time.time() - start_time:.2f} seconds")

# =============================================================================
# PART 2: MULTIPROCESSING
# =============================================================================

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

print("\n" + "=" * 60)
print("MULTIPROCESSING EXAMPLES")
print("=" * 60)


# Example 5: Basic Multiprocessing
# --------------------------------
def cpu_intensive_task(n):
    """CPU-intensive task - calculating sum of squares"""
    process_id = os.getpid()
    result = sum(i * i for i in range(n))
    return f"Process {process_id}: Sum of squares up to {n} = {result}"


print("\n5. Basic Multiprocessing:")

if __name__ == "__main__":  # Required for multiprocessing on Windows
    # Create processes manually
    processes = []
    numbers = [100000, 200000, 150000]

    start_time = time.time()

    for num in numbers:
        p = multiprocessing.Process(
            target=lambda x: print(cpu_intensive_task(x)), args=(num,)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print(f"All processes completed in {time.time() - start_time:.2f} seconds")

# Example 6: Process Pool with Return Values
# ------------------------------------------
print("\n6. Process Pool with Return Values:")


def calculate_factorial(n):
    """Calculate factorial (CPU-intensive)"""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return f"Factorial of {n} = {result}"


if __name__ == "__main__":
    numbers = [10, 15, 12, 8, 20]

    start_time = time.time()

    # Using ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks and get futures
        futures = [executor.submit(calculate_factorial, n) for n in numbers]

        # Collect results
        for future in futures:
            print(future.result())

    print(f"All calculations completed in {time.time() - start_time:.2f} seconds")

# Example 7: Sharing Data Between Processes
# -----------------------------------------
print("\n7. Sharing Data Between Processes:")


def worker_with_shared_data(shared_list, shared_value, lock, worker_id):
    """Worker that modifies shared data"""
    with lock:
        shared_value.value += 1
        shared_list.append(f"Worker-{worker_id}")
        print(f"Worker {worker_id}: Added to shared data")


if __name__ == "__main__":
    # Create shared data structures
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    shared_value = multiprocessing.Value("i", 0)  # 'i' for integer
    lock = multiprocessing.Lock()

    processes = []
    for i in range(5):
        p = multiprocessing.Process(
            target=worker_with_shared_data, args=(shared_list, shared_value, lock, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Final shared list: {list(shared_list)}")
    print(f"Final shared value: {shared_value.value}")

# Example 8: Inter-Process Communication with Queue
# ------------------------------------------------
print("\n8. Inter-Process Communication with Queue:")


def producer_process(q, num_items):
    """Producer process"""
    for i in range(num_items):
        item = f"Item-{i}"
        q.put(item)
        print(f"Produced: {item}")
        time.sleep(0.1)


def consumer_process(q, process_name):
    """Consumer process"""
    while True:
        try:
            item = q.get(timeout=2)  # Wait 2 seconds for item
            print(f"Consumer {process_name} processed: {item}")
            time.sleep(0.2)
        except:
            print(f"Consumer {process_name} timed out, exiting")
            break


if __name__ == "__main__":
    # Create a multiprocessing queue
    mp_queue = multiprocessing.Queue()

    # Create producer process
    producer = multiprocessing.Process(target=producer_process, args=(mp_queue, 5))

    # Create consumer processes
    consumer1 = multiprocessing.Process(target=consumer_process, args=(mp_queue, "C1"))
    consumer2 = multiprocessing.Process(target=consumer_process, args=(mp_queue, "C2"))

    # Start all processes
    producer.start()
    consumer1.start()
    consumer2.start()

    # Wait for producer to finish
    producer.join()

    # Let consumers finish processing
    consumer1.join()
    consumer2.join()

# =============================================================================
# PART 3: PERFORMANCE COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)


def cpu_bound_task(n):
    """CPU-intensive task for comparison"""
    return sum(i * i for i in range(n))


def io_bound_task():
    """I/O-intensive task simulation"""
    time.sleep(0.5)  # Simulate I/O delay
    return "I/O task completed"


def compare_performance():
    """Compare threading vs multiprocessing performance"""

    print("\n9. Performance Comparison:")

    # CPU-bound task comparison
    print("\nCPU-bound tasks (calculating sum of squares):")
    numbers = [100000] * 4

    # Sequential execution
    start_time = time.time()
    sequential_results = [cpu_bound_task(n) for n in numbers]
    sequential_time = time.time() - start_time
    print(f"Sequential: {sequential_time:.2f} seconds")

    # Threading (not ideal for CPU-bound)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        thread_results = list(executor.map(cpu_bound_task, numbers))
    thread_time = time.time() - start_time
    print(f"Threading: {thread_time:.2f} seconds")

    # Multiprocessing (ideal for CPU-bound)
    if __name__ == "__main__":
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            process_results = list(executor.map(cpu_bound_task, numbers))
        process_time = time.time() - start_time
        print(f"Multiprocessing: {process_time:.2f} seconds")

    print(f"\nFor CPU-bound tasks:")
    print(f"Threading is {thread_time / sequential_time:.2f}x sequential speed")
    print(f"Multiprocessing is {process_time / sequential_time:.2f}x sequential speed")

    # I/O-bound task comparison
    print("\nI/O-bound tasks (simulated network requests):")
    tasks = [io_bound_task] * 4

    # Sequential execution
    start_time = time.time()
    sequential_results = [task() for task in tasks]
    sequential_time = time.time() - start_time
    print(f"Sequential: {sequential_time:.2f} seconds")

    # Threading (ideal for I/O-bound)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        thread_results = list(executor.map(lambda f: f(), tasks))
    thread_time = time.time() - start_time
    print(f"Threading: {thread_time:.2f} seconds")

    print(f"\nFor I/O-bound tasks:")
    print(f"Threading is {sequential_time / thread_time:.2f}x faster than sequential")


if __name__ == "__main__":
    compare_performance()

# =============================================================================
# BEST PRACTICES AND GUIDELINES
# =============================================================================

print("\n" + "=" * 60)
print("BEST PRACTICES")
print("=" * 60)

"""
WHEN TO USE WHAT:
=================

USE THREADING FOR:
- I/O-bound tasks (file operations, network requests, database queries)
- Tasks that spend time waiting for external resources
- UI applications to keep interface responsive
- When you need shared state between concurrent operations

USE MULTIPROCESSING FOR:
- CPU-bound tasks (mathematical calculations, data processing, image processing)
- Tasks that can be parallelized and don't need shared state
- When you need to bypass Python's GIL
- Fault isolation (one process crash doesn't affect others)

KEY CONSIDERATIONS:
==================

THREADING:
✓ Shared memory space (easy data sharing)
✓ Lower overhead
✓ Good for I/O-bound tasks
✗ GIL limits CPU-bound performance
✗ Race conditions possible
✗ Debugging can be challenging

MULTIPROCESSING:
✓ True parallelism for CPU-bound tasks
✓ Process isolation (fault tolerance)
✓ No GIL limitations
✗ Higher memory overhead
✗ Complex inter-process communication
✗ Slower to create processes

COMMON PATTERNS:
===============

1. Producer-Consumer: Use Queue for thread-safe communication
2. Worker Pool: Use ThreadPoolExecutor/ProcessPoolExecutor
3. Divide and Conquer: Split large tasks across processes
4. Pipeline: Chain operations with queues

SYNCHRONIZATION TOOLS:
=====================

Threading:
- Lock: Mutual exclusion
- RLock: Reentrant lock
- Semaphore: Limit resource access
- Event: Simple signaling
- Condition: Complex waiting conditions

Multiprocessing:
- Lock: Process-safe mutual exclusion  
- Queue: Inter-process communication
- Pipe: Bidirectional communication
- Manager: Shared objects
- Value/Array: Shared memory

DEBUGGING TIPS:
==============

1. Use logging instead of print for thread-safe output
2. Always use context managers (with statements) for locks
3. Avoid global variables in multiprocessing
4. Handle exceptions properly in worker functions
5. Use timeouts to avoid deadlocks
6. Test with different numbers of workers
"""

print(
    "Study complete! Practice these examples and experiment with different scenarios."
)
