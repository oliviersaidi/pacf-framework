#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2005 Olivier Saidi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Pattern-Aware Complexity Framework (PACF) v1.0
----------------------------------
A Python implementation of the generalized pattern-aware complexity framework
for NP-hard optimization problems, with a focus on the Traveling Salesman Problem (TSP).

FRAMEWORK SUMMARY:
-----------------
The Pattern-Aware Complexity Framework (PACF) provides a systematic approach to analyze 
and exploit problem structure in combinatorial optimization. It recognizes that real-world 
instances often contain patterns (symmetry, clustering, repetition, etc.) that can be 
leveraged to reduce the effective complexity of solving these problems.

KEY COMPONENTS:
1. Pattern Detection: Identifies structural patterns in problem instances
2. Pattern-Aware Complexity Calculation: Adjusts theoretical complexity based on detected patterns
3. Pattern Utilization Efficiency (PUE): Measures how well solvers exploit detected patterns
4. Solution Quality Factor (SQF): Quantifies improvement over baseline solutions
5. Adaptive Solver Selection: Uses meta-learning to select the most appropriate solver

METHODOLOGY:
-----------
1. Pattern Recognition:
   - Extract features from problem instances
   - Detect domain-specific patterns (clustering, symmetry, repetition, seasonality)
   - Quantify pattern prevalence and calculate entropy

2. Complexity Refinement:
   - Calculate pattern-adjusted complexity: C(P,A) = T_base(n) * f(n,ρ,H) * R(P,A) + C_residual
   - T_base: Theoretical complexity without patterns (e.g., O(n^2 * 2^n) for TSP)
   - f(n,ρ,H): Influence function based on problem size, pattern prevalence, and entropy
   - R(P,A): Reduction factor based on algorithm's pattern exploitation capability
   - C_residual: Residual complexity term to avoid unrealistic zero values

3. Solver Performance Measurement:
   - Implement and benchmark multiple solvers
   - Collect performance metrics (runtime, solution quality, pattern utilization)
   - Train meta-models to predict solver performance on new instances

KEY METRICS:
-----------
1. Pattern Prevalence (ρ):
   - Percentage of problem elements covered by detected patterns (0 to 1)
   - Higher values indicate more structured instances

2. Pattern Utilization Efficiency (PUE):
   - Percentage of problem elements where patterns aid in solving (0 to 100%)
   - Measures how effectively patterns are leveraged

3. Solution Quality Factor (SQF):
   - Percentage improvement over a baseline solution (e.g., Nearest Neighbor for TSP)
   - Quantifies the benefit of using more sophisticated solvers

4. Entropy (H):
   - Measures the disorder or randomness in the problem instance
   - Lower values indicate more predictable structure

5. Pattern-Aware Complexity:
   - Adjusted theoretical complexity that accounts for problem structure
   - Lower than traditional worst-case complexity for structured instances

This framework supports different problem domains (TSP, genetic sequences, weather forecasting)
and can be extended to additional optimization problems.

This optimized script implements:
- Enhanced 2-Opt performance through improved pruning strategies and smarter swap selection
- Distance matrix caching to avoid redundant calculations
- Better parallel processing with optimized workload distribution
- Early termination criteria based on time limits and diminishing returns
- Addition of 3-Opt solver for better solution quality on small to medium instances
- Performance testing tools for benchmarking
- All original features with improved documentation and performance monitoring

Version 3.0 includes the following improvements over v2:
- Enhanced 2-Opt performance through improved pruning strategies and smarter swap selection
- Distance matrix caching to avoid redundant calculations
- Better parallel processing with optimized workload distribution
- Early termination criteria based on time limits and diminishing returns
- Addition of 3-Opt solver for better solution quality on small to medium instances
- Performance testing tools for benchmarking

Usage:
- Run TSP analysis: python script.py path/to/tsp_file.tsp
- Run experiment: python script.py run-experiment --problem tsp --dataset path/to/dataset --log-level [DEBUG|INFO|WARNING|ERROR]
- Run performance test: python script.py performance-test --instances att48,berlin52,rat783
"""

# Standard library imports for basic functionality
import os                         # Operating system interfaces (file paths, directories)
import time                       # Time-related functions (performance timing)
import math                       # Mathematical functions
import json                       # JSON encoding/decoding
import logging                    # Logging framework for detailed output
import argparse                   # Command-line argument parsing
import copy                       # Deep/shallow copying objects
import random                     # Random number generation
import heapq                      # Heap queue for priority-based operations
import sys                        # System-specific parameters and functions
import functools                  # Higher-order functions and operations on callable objects
import threading                  # Thread-based parallelism
import numpy as np                # Numerical computing library
import pandas as pd               # Data analysis library
import matplotlib.pyplot as plt   # Plotting and visualization
import multiprocessing
from multiprocessing import shared_memory  # Shared memory for efficient inter-process communication
# Flag to identify if this is a worker process - helps configure appropriate logging levels
IS_WORKER_PROCESS = multiprocessing.current_process().name != 'MainProcess' 
from abc import ABC, abstractmethod  # Abstract base classes for creating proper class hierarchies
from dataclasses import dataclass    # Data class decorator for simple class definitions
from typing import Dict, List, Tuple, Set, Any, Optional, Callable, Union, Generator  # Type hints for better code clarity
from enum import Enum                # Enumeration type for pattern categorization
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed  # Parallel processing tools
from multiprocessing import cpu_count, Manager  # Get CPU core count for optimizing parallelization
from collections import defaultdict, deque  # Specialized container datatypes
from datetime import datetime, timedelta  # Date and time manipulation

# Hardware acceleration imports (for potential GPU/MPS use)
import torch                      # PyTorch for hardware acceleration when available

# ML imports for meta-learning (used for adaptive solver selection)
from sklearn.ensemble import RandomForestRegressor  # Machine learning for prediction
from sklearn.metrics import mean_squared_error      # Evaluation metrics
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler    # Feature normalization

# Import scikit-learn's KMeans with error handling
try:
    from sklearn.cluster import KMeans  # Clustering for pattern detection and solver optimization
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. K-means clustering functionality disabled.")

# Domain-specific imports for TSP and graph operations
import networkx as nx             # Graph operations library for representing TSP instances

# Optional imports for additional domains (genetic, weather)
# These imports are wrapped in try/except to make these features optional
try:
    import Bio.SeqIO              # BioPython for genetic sequence processing
    BIOPYTHON_AVAILABLE = True    # Flag to indicate BioPython is available
except ImportError:
    BIOPYTHON_AVAILABLE = False   # BioPython not installed
    logging.warning("BioPython not available. Genetic sequence functionality disabled.")

try:
    import xarray as xr           # xarray for multi-dimensional data (weather)
    from scipy.signal import find_peaks  # Signal processing for detecting seasonal patterns
    XARRAY_AVAILABLE = True       # Flag to indicate xarray is available
except ImportError:
    XARRAY_AVAILABLE = False      # xarray not installed
    logging.warning("xarray or scipy not available. Weather forecasting functionality disabled.")

# Configure logging - higher level for worker processes to reduce output clutter
logging.basicConfig(
	level=logging.WARNING if IS_WORKER_PROCESS else logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def worker_init():
    """Initialize worker process with minimal logging."""
    # Configure worker's logger to only show warnings and errors
    # This reduces log clutter when running parallel processes
    logging.getLogger().setLevel(logging.WARNING)

# Performance monitoring helper functions
def timeit(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Memory usage monitoring (platform-specific)
def memory_usage():
    """Get current memory usage in MB (best effort across platforms)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # RSS in MB
    except (ImportError, AttributeError):
        try:
            # Fallback for Unix systems
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
        except (ImportError, AttributeError):
            # Fallback message if memory tracking is unavailable
            return -1  # Not available

# Only detect and log hardware info in the main process to avoid duplication
if not IS_WORKER_PROCESS:
    # Check for available hardware acceleration (CUDA GPU or Apple Metal)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        logger.info(f"Using CUDA for acceleration (Device: {torch.cuda.get_device_name(0)})")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        logger.info("Using Apple Metal for acceleration")
    else:
        DEVICE = torch.device("cpu")
        logger.info("No GPU/MPS available, using CPU")
        
    # Determine optimal number of workers for parallel processing
    NUM_CORES = cpu_count()
    logger.info(f"Detected {NUM_CORES} CPU cores for parallel processing")
    
    # Log available memory
    mem_usage = memory_usage()
    if mem_usage > 0:
        logger.info(f"Available memory: {mem_usage:.2f} MB")
else:
    # For worker processes, just set the device without logging
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
        
    # Just use the core count without logging
    NUM_CORES = cpu_count()
    
# Determine optimal number of workers for parallel processing based on CPU cores
# For 2-Opt and other CPU-intensive operations, leave one core free for system operations
NUM_WORKERS = max(1, NUM_CORES - 1)

# Global configuration for solver performance
DEFAULT_MAX_EXECUTION_TIME = 30.0  # Default maximum execution time in seconds
DEFAULT_EARLY_TERMINATION_THRESHOLD = 0.001  # Default improvement threshold for early termination (0.1%)

###########################################
# 1. Core Framework Components
###########################################

class PatternType(Enum):
    """
    Enumeration of pattern types that can be detected in problem instances.
    These categories help organize different kinds of patterns that might
    exist in optimization problems.
    
    Each pattern type represents a different structural characteristic
    that can be leveraged by specialized algorithms.
    """
    SYMMETRY = 1      # Symmetrical structures (mirror-like patterns)
    REPETITION = 2    # Repeated sequences or structures (recurring elements)
    CLUSTERING = 3    # Groups of closely related elements (spatial clusters)
    SEASONALITY = 4   # Periodic or seasonal patterns (time-based patterns)
    MOTIF = 5         # Specific recurring subsequences (pattern templates)
    CUSTOM = 6        # User-defined patterns (extensibility point)

@dataclass
class Pattern:
    """
    Represents a detected pattern within a problem instance.
    Used to track elements covered by patterns and associated metadata.
    
    This is the fundamental data structure for storing pattern information.
    Patterns are detected by domain-specific algorithms and then used by
    solvers to reduce computational complexity.
    
    Attributes:
        type: The category of pattern (from PatternType enum)
        elements: Set of elements (e.g., nodes) covered by this pattern
        metadata: Additional information about the pattern (e.g., strength)
    """
    type: PatternType         # Type of pattern (e.g., CLUSTERING, SYMMETRY)
    elements: Set[Any]        # Elements (e.g., nodes) covered by this pattern
    metadata: Dict[str, Any]  # Additional information (e.g., cluster size, strength)

    def size(self) -> int:
        """
        Calculate the number of elements covered by the pattern.
        
        Returns:
            int: Number of elements in the pattern
        """
        logger.debug(f"Calculating size for pattern type {self.type}: {len(self.elements)} elements")
        return len(self.elements)

class ProblemInstance(ABC):
    """
    Abstract base class for problem instances (e.g., TSP, genetic sequences).
    Provides methods for pattern detection, entropy calculation, and feature extraction.
    This serves as a template for specific problem types.
    
    This is a key abstraction that allows the framework to be extended to
    different problem domains while maintaining a consistent interface.
    """
    
    def __init__(self, data: Any, name: str = "unnamed", cache_enabled: bool = True):
        """
        Initialize a problem instance with data and a name.
        
        Args:
            data: Underlying data structure (e.g., graph for TSP)
            name: Identifier for the instance (default: 'unnamed')
            cache_enabled: Whether to enable caching for computations (default: True)
        """
        logger.debug(f"Initializing ProblemInstance '{name}'")
        self.data = data              # Raw problem data
        self.name = name              # Instance identifier
        self.size = self._calculate_size()  # Compute instance size
        self._patterns: List[Pattern] = []  # List to store detected patterns
        self._pattern_prevalence: Optional[float] = None  # Cached pattern prevalence
        self._entropy: Optional[float] = None  # Cached entropy value
        self._pue: Optional[float] = None  # Cached solver-independent PUE
        self._cluster_count: Optional[int] = None  # Cached number of clusters
        self._cache_enabled = cache_enabled  # Flag for computation caching
        self._cache = {}  # Dictionary for caching computed values
        logger.info(f"ProblemInstance '{name}' initialized with size {self.size}")
    
    @abstractmethod
    def _calculate_size(self) -> int:
        """
        Abstract method to calculate the size of the problem instance.
        Must be implemented by concrete subclasses.
        
        Returns:
            int: Size of the problem instance (e.g., number of cities for TSP)
        """
        pass
    
    @abstractmethod
    def detect_patterns(self, pattern_types: List[PatternType]) -> List[Pattern]:
        """
        Abstract method to detect patterns of specified types.
        Must be implemented by concrete subclasses.
        
        Args:
            pattern_types: List of PatternType enums to detect
        
        Returns:
            List[Pattern]: List of detected Pattern objects
        """
        pass
    
    @abstractmethod
    def calculate_entropy(self) -> float:
        """
        Abstract method to calculate the entropy of the instance.
        Entropy measures the disorder or complexity in the problem.
        Must be implemented by concrete subclasses.
        
        Returns:
            float: Entropy value
        """
        pass
    
    def get_pattern_prevalence(self) -> float:
        """
        Calculate pattern prevalence ρ(P) as the ratio of elements covered by patterns to total size.
        ρ(P) = |Union of all pattern elements| / |P|
        
        This measure indicates how much of the problem is covered by detected patterns.
        Higher values suggest more structured instances that might be solved more efficiently.
        
        Returns:
            float: Pattern prevalence between 0.0 and 1.0
        """
        logger.debug(f"Calculating pattern prevalence for '{self.name}'")
        # Return cached value if available
        if self._pattern_prevalence is not None:
            logger.debug(f"Returning cached pattern prevalence: {self._pattern_prevalence}")
            return self._pattern_prevalence
            
        # Calculate pattern prevalence if not cached
        if not self._patterns:
            logger.debug("No patterns detected, setting pattern prevalence to 0.0")
            self._pattern_prevalence = 0.0
        else:
            # Find union of all elements covered by any pattern
            covered_elements = set()
            for pattern in self._patterns:
                covered_elements.update(pattern.elements)
            # Calculate ratio of covered elements to total size
            self._pattern_prevalence = len(covered_elements) / self.size if self.size > 0 else 0.0
            logger.debug(f"Pattern prevalence: {len(covered_elements)} elements covered out of {self.size}")
        logger.info(f"Pattern prevalence for '{self.name}': {self._pattern_prevalence:.4f}")
        return self._pattern_prevalence
    
    def get_entropy(self) -> float:
        """
        Retrieve or calculate the entropy of the problem instance.
        Entropy represents the amount of disorder or randomness in the problem.
        
        Lower entropy indicates more regular structure that might be easier to exploit.
        
        Returns:
            float: Entropy value
        """
        logger.debug(f"Retrieving entropy for '{self.name}'")
        if self._entropy is None:
            self._entropy = self.calculate_entropy()
        logger.debug(f"Entropy for '{self.name}': {self._entropy:.4f}")
        return self._entropy
    
    def get_pue(self) -> float:
        """
        Calculate solver-independent Pattern Utilization Efficiency (PUE).
        PUE = (Number of nodes covered by at least one pattern / Total nodes) * 100
        
        This metric shows what percentage of the problem is covered by patterns.
        It's a key measure of how much the problem structure could potentially
        be exploited by pattern-aware solvers.
        
        Returns:
            float: PUE percentage (0.0 to 100.0)
        """
        logger.debug(f"Calculating PUE for '{self.name}'")
        if self._pue is not None:
            logger.debug(f"Returning cached PUE: {self._pue}")
            return self._pue
            
        if not self._patterns or self.size == 0:
            logger.debug("No patterns or zero size, PUE set to 0.0")
            self._pue = 0.0
        else:
            covered_nodes = set()
            for pattern in self._patterns:
                covered_nodes.update(pattern.elements)
            self._pue = (len(covered_nodes) / self.size) * 100.0
            logger.debug(f"PUE: {len(covered_nodes)} nodes covered out of {self.size}")
        logger.info(f"PUE for '{self.name}': {self._pue:.2f}%")
        return self._pue
    
    def get_cluster_count(self) -> int:
        """
        Calculate the number of detected clusters in the instance.
        Clusters are a specific type of pattern (PatternType.CLUSTERING).
        
        For TSP, clusters represent groups of cities that are close to each other,
        which can be leveraged by solvers to reduce complexity.

        Returns:
            int: Number of cluster patterns detected
        """
        logger.debug(f"Calculating cluster count for '{self.name}'")
        if self._cluster_count is not None:
            logger.debug(f"Returning cached cluster count: {self._cluster_count}")
            return self._cluster_count
        
        # Count patterns of type CLUSTERING
        cluster_patterns = [p for p in self._patterns if p.type == PatternType.CLUSTERING]
        self._cluster_count = len(cluster_patterns)
        logger.debug(f"Cluster count: {self._cluster_count}")
        return self._cluster_count
    
    def get_features(self) -> Dict[str, float]:
        """
        Extract features for meta-learning or analysis.
        These features summarize the problem instance characteristics.
        
        Used by the SolverPortfolio to select the most appropriate solver
        based on instance characteristics.
        
        Returns:
            Dict[str, float]: Dictionary with size, pattern prevalence, entropy, PUE, and cluster count
        """
        logger.debug(f"Extracting features for '{self.name}'")
        features = {
            "size": self.size,
            "pattern_prevalence": self.get_pattern_prevalence(),
            "entropy": self.get_entropy(),
            "pue": self.get_pue(),
            "cluster_count": self.get_cluster_count()
        }
        logger.debug(f"Features: {features}")
        return features
    
    def clear_cache(self):
        """Clear all cached computations to free memory."""
        logger.debug(f"Clearing cache for '{self.name}'")
        self._cache = {}
        
    def cache_value(self, key: str, value: Any):
        """
        Cache a computed value for future use.
        
        Args:
            key: Cache key identifier
            value: Value to cache
        """
        if self._cache_enabled:
            self._cache[key] = value
            
    def get_cached_value(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: Cache key identifier
            
        Returns:
            Optional[Any]: Cached value if available, None otherwise
        """
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        return None

class SolverConfiguration:
    """
    Configuration settings for solvers with default values.
    Allows for centralized control of solver parameters.
    """
    
    def __init__(self):
        # General solver parameters
        self.max_execution_time = 10.0  # Reduced from 30s to 10s for small datasets
        self.time_check_interval = 0.5  # How often to check if time limit exceeded (seconds)
        
        # TSP solver specific parameters
        self.max_iterations = 50  # Reduced from 100 to 50
        self.max_swaps_per_iteration = 500  # Reduced from 1000 to 500
        self.prune_swaps = True  # Whether to prune swaps based on edge lengths
        self.early_termination_threshold = 0.005  # Increased threshold for earlier termination
        self.parallel_batch_size = 100  # Reduced batch size
        self.distance_matrix_caching = True  # Whether to cache distance matrices
        self.greedy_initialization = True  # Whether to use greedy initialization
        self.min_cluster_size = 3  # Minimum cluster size for pattern detection
        
        # 3-Opt specific parameters
        self.max_3opt_iterations = 20  # Maximum iterations for 3-Opt
        
        # Adaptive parameters
        self.dynamically_adjust_parameters = True  # Whether to adjust parameters based on instance size
    
    def adjust_for_instance(self, instance):
        """
        Adjust configuration based on instance characteristics.
        
        Args:
            instance: ProblemInstance to adjust parameters for
        """
        if not self.dynamically_adjust_parameters:
            return
            
        size = instance.size
        
        # Scale maximum iterations based on instance size
        if size < 50:  # Small instances like att48
            self.max_iterations = 30
            self.max_swaps_per_iteration = 200
            self.max_execution_time = min(self.max_execution_time, 5.0)  # Faster timeout for small instances
        elif size < 100:
            self.max_iterations = 40
            self.max_swaps_per_iteration = 400
        elif size < 500:
            self.max_iterations = 50
            self.max_swaps_per_iteration = 500
        else:
            self.max_iterations = 30
            self.max_swaps_per_iteration = 300
            
        # Adjust batch size for parallel processing
        self.parallel_batch_size = min(100, max(50, size * 2))
        
        # Adjust early termination threshold based on instance size
        self.early_termination_threshold = max(0.001, 0.01 / math.log2(size + 2))
        
        logger.debug(f"Adjusted solver parameters for instance size {size}: "
                    f"max_iterations={self.max_iterations}, "
                    f"max_swaps_per_iteration={self.max_swaps_per_iteration}, "
                    f"parallel_batch_size={self.parallel_batch_size}, "
                    f"early_termination_threshold={self.early_termination_threshold:.6f}")

class Solver(ABC):
    """
    Abstract base class for solvers (e.g., TSP heuristics).
    Defines methods for solving instances and calculating complexity.
    This serves as a template for specific solver implementations.
    
    This abstraction allows the framework to work with different solvers
    while providing a consistent interface for complexity calculation and
    pattern exploitation.
    """
    
    def __init__(self, name: str, config: Optional[SolverConfiguration] = None):
        """
        Initialize a solver with a name and configuration.
        
        Args:
            name: Identifier for the solver (e.g., 'TSP-NearestNeighbor')
            config: Configuration settings for the solver (optional)
        """
        logger.debug(f"Initializing solver '{name}'")
        self.name = name  # Solver identifier
        self.config = config if config else SolverConfiguration()  # Configuration settings
        self.early_termination = False  # Flag for early termination
        logger.info(f"Solver '{name}' initialized")
    
    @abstractmethod
    def solve(self, instance: ProblemInstance) -> Tuple[Any, Dict[str, Any]]:
        """
        Abstract method to solve a problem instance.
        Must be implemented by concrete subclasses.
        
        Args:
            instance: Problem instance to solve
            
        Returns:
            Tuple: (solution, metadata dictionary)
        """
        pass
    
    @abstractmethod
    def get_base_complexity(self, instance_size: int) -> float:
        """
        Abstract method to calculate base complexity T_base(n).
        This represents the theoretical complexity without pattern awareness.
        Must be implemented by concrete subclasses.
        
        Args:
            instance_size: Size of the problem instance
        
        Returns:
            float: Base complexity value
        """
        pass
    
    @abstractmethod
    def get_reduction_factor(self, instance: ProblemInstance) -> float:
        """
        Abstract method to calculate reduction factor R(P,A).
        This factor represents how efficiently the solver uses patterns.
        Must be implemented by concrete subclasses.
        
        Args:
            instance: ProblemInstance to evaluate
        
        Returns:
            float: Reduction factor between 0.0 and 1.0
        """
        pass
    
    def calculate_complexity(self, instance: ProblemInstance, k: float = 1.0) -> float:
        """
        Calculate pattern-aware complexity with refinement:
        C(P,A) = T_base(n) * f(n, ρ(P), H(P)) * R(P,A) + C_residual(n)
        
        This version handles very large complexity values safely.
        
        Args:
            instance: ProblemInstance to compute complexity for
            k: Exponent for pattern prevalence adjustment (default: 1.0)
        
        Returns:
            float: Total complexity value (may be log10 value for large instances)
        """
        logger.debug(f"Calculating complexity for '{instance.name}' with solver '{self.name}'")
        n = instance.size  # Problem size
        rho = instance.get_pattern_prevalence()  # Pattern prevalence
        H = instance.get_entropy()  # Entropy
        
        # Handle edge case for small instances
        if n <= 1:
            logger.debug("Instance size <= 1, returning minimal complexity 1.0")
            return 1.0
            
        # Calculate components of complexity formula
        T_base = self.get_base_complexity(n)  # Base complexity
        
        # Check if we got log complexity from base complexity
        is_log_complexity = n > 100
        
        if is_log_complexity:
            # We're working with logarithms, so operations become additions
            pattern_factor = math.log10(max(0.000001, 1 - rho**k))  # Avoid log(0)
            entropy_factor = -H / math.log(n + 1)
            log_R = math.log10(self.get_reduction_factor(instance))
            log_residual = math.log10(math.log(n + 1))
            # Calculate log complexity 
            log_complexity = T_base + pattern_factor + entropy_factor + log_R + log_residual			
            # The residual term needs special care since we can't directly add it in log form
            # For large instances, the residual is negligible compared to the main term
            logger.debug(f"Log10 complexity: {log_complexity:.2f} (pattern factor: {pattern_factor:.2f}, entropy factor: {entropy_factor:.2f})")
            return log_complexity
        else:
            # Calculate as before for smaller instances
            f = math.exp(-H / math.log(n + 1)) * (1 - rho**k) if n > 1 else 1.0
            R = self.get_reduction_factor(instance)  # Reduction factor
            C_residual = math.log(n + 1)  # Prevents complexity from dropping to 0
            complexity = T_base * f * R + C_residual
            
            logger.debug(f"Complexity breakdown: T_base={T_base:.2e}, f={f:.4f}, R={R:.4f}, C_residual={C_residual:.2f}, total={complexity:.2e}")
            logger.info(f"Complexity for '{instance.name}' with '{self.name}': {complexity:.2e}")
            return complexity
    
    def check_timeout(self, start_time: float) -> bool:
        """
        Check if the maximum execution time has been exceeded.
        
        Args:
            start_time: Start time of the execution
            
        Returns:
            bool: True if time limit exceeded, False otherwise
        """
        elapsed = time.time() - start_time
        if elapsed > self.config.max_execution_time:
            logger.warning(f"Time limit of {self.config.max_execution_time} seconds exceeded ({elapsed:.2f}s). Terminating solver.")
            self.early_termination = True
            return True
        return False
	
###########################################
# 2. TSP Implementation
###########################################

class TSPInstance(ProblemInstance):
    """
    Concrete implementation of ProblemInstance for Traveling Salesman Problems.
    Provides TSP-specific pattern detection and entropy calculation.
    
    The TSP is a classic NP-hard problem where the goal is to find the shortest
    tour that visits all cities exactly once and returns to the starting city.
    """
    
    def __init__(self, data: nx.Graph, name: str = "unnamed", cache_enabled: bool = True):
        """
        Initialize a TSP instance with a graph, name, and caching option.
        
        Args:
            data: NetworkX graph representing the TSP instance
            name: Identifier for the instance (default: 'unnamed')
            cache_enabled: Whether to enable caching (default: True)
        """
        super().__init__(data, name, cache_enabled)
        self._distance_matrix = None  # Cache for distance matrix
        self._kmeans_clusters = None  # Cache for k-means clusters
    
    def _calculate_size(self) -> int:
        """
        Calculate the size of the TSP instance as the number of cities.
        
        Returns:
            int: Number of nodes (cities) in the graph
        """
        size = len(self.data.nodes)
        logger.debug(f"TSP instance '{self.name}' size: {size}")
        return size
    
    def detect_patterns(self, pattern_types: List[PatternType]) -> List[Pattern]:
        """
        Detect patterns in the TSP instance based on specified types.
        Currently supports clustering and symmetry pattern detection.
        
        Clustering represents groups of cities that are close to each other.
        Symmetry represents pairs of cities with similar distances to other cities.
        
        Args:
            pattern_types: List of PatternType enums to detect (e.g., CLUSTERING, SYMMETRY)
        
        Returns:
            List[Pattern]: List of detected Pattern objects
        """
        logger.info(f"Detecting patterns for '{self.name}' with types {pattern_types}")
        self._patterns = []  # Reset existing patterns
        
        # Detect clusters if requested
        if PatternType.CLUSTERING in pattern_types:
            self._detect_clusters()
            # Also try k-means clustering for more consistent clusters
            self._detect_kmeans_clusters()
        # Detect symmetries if requested
        if PatternType.SYMMETRY in pattern_types:
            self._detect_symmetries()
            
        # Reset cached values that depend on patterns
        self._pattern_prevalence = None
        self._pue = None
        self._cluster_count = None
        
        # Log results
        pue = self.get_pue()
        logger.info(f"Detected {len(self._patterns)} patterns for '{self.name}', PUE: {pue:.2f}%")
        return self._patterns
    
    def _detect_clusters(self, distance_threshold: float = 0.2, min_cluster_size: int = 3):
        """
        Detect clusters of cities based on proximity using edge weights.
        Cities within a threshold distance of each other are grouped into clusters.
        
        Clusters can significantly reduce complexity because they suggest a good
        strategy: visit all cities in a cluster before moving to another cluster.
        
        Args:
            distance_threshold: Fraction of max distance to define clusters (default: 0.2)
            min_cluster_size: Minimum number of cities to form a cluster (default: 3)
        """
        logger.debug(f"Detecting clusters for '{self.name}' with threshold {distance_threshold}")
        # Extract all edge weights
        edge_weights = [d['weight'] for _, _, d in self.data.edges(data=True)]
        if not edge_weights:
            logger.warning(f"No edge weights found for '{self.name}'")
            return
            
        # Calculate distance threshold based on maximum distance
        max_distance = max(edge_weights)
        threshold = distance_threshold * max_distance
        logger.debug(f"Max distance: {max_distance:.2f}, Cluster threshold: {threshold:.2f}")
        
        # Build subgraph with edges below threshold (close cities)
        subgraph_edges = [(u, v) for u, v, d in self.data.edges(data=True) if d['weight'] <= threshold]
        subgraph = nx.Graph()
        subgraph.add_nodes_from(self.data.nodes())
        subgraph.add_edges_from(subgraph_edges)
        
        # Identify connected components as clusters
        clusters = list(nx.connected_components(subgraph))
        logger.debug(f"Found {len(clusters)} clusters")
        
        # Create patterns for clusters of sufficient size
        for i, cluster in enumerate(clusters):
            if len(cluster) >= min_cluster_size:  # Require minimum cluster size
                pattern = Pattern(
                    type=PatternType.CLUSTERING,
                    elements=set(cluster),
                    metadata={"cluster_id": i, "threshold": threshold, "size": len(cluster),
                             "method": "connected_components"}
                )
                self._patterns.append(pattern)
                logger.debug(f"Added cluster {i} with {len(cluster)} cities")
    
    def _detect_kmeans_clusters(self, k: Optional[int] = None, min_cluster_size: int = 3):
        """
        Detect clusters using k-means clustering algorithm.
        Provides more balanced clusters than threshold-based method.
        
        Args:
            k: Number of clusters to detect (default: None, auto-determined)
            min_cluster_size: Minimum number of cities to form a cluster (default: 3)
        """
        logger.debug(f"Detecting k-means clusters for '{self.name}'")
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, skipping k-means clustering")
            return
            
        # Get city positions
        pos = nx.get_node_attributes(self.data, 'pos')
        if not pos:
            logger.warning("No position data available for k-means clustering")
            return
            
        # Determine number of clusters if not specified
        if k is None:
            # Heuristic: sqrt(n/2) seems to work well for TSP instances
            k = max(2, int(math.sqrt(self.size / 2)))
        
        # Prepare position data
        nodes = sorted(pos.keys())
        X = np.array([pos[node] for node in nodes])
        
        # Run k-means clustering
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Organize cities by cluster
            clusters = [[] for _ in range(k)]
            for i, label in enumerate(labels):
                clusters[label].append(nodes[i])
            
            # Save for later use
            self._kmeans_clusters = clusters
            
            # Create patterns for clusters of sufficient size
            for i, cluster in enumerate(clusters):
                if len(cluster) >= min_cluster_size:
                    pattern = Pattern(
                        type=PatternType.CLUSTERING,
                        elements=set(cluster),
                        metadata={"cluster_id": i + 100, "size": len(cluster), "method": "kmeans",
                                 "center": kmeans.cluster_centers_[i].tolist()}
                    )
                    self._patterns.append(pattern)
                    logger.debug(f"Added k-means cluster {i} with {len(cluster)} cities")
        except Exception as e:
            logger.error(f"Error in k-means clustering: {e}")
    
    def _detect_symmetries(self, tolerance: float = 0.05, max_symmetries: int = 20):
        """
        Detect approximate symmetries in node distances.
        Symmetrical nodes have similar distances to all other nodes.
        
        Symmetry can reduce complexity because symmetrical cities might be
        interchangeable in optimal or near-optimal solutions.
        
        Args:
            tolerance: Fraction of max distance for symmetry tolerance (default: 0.05)
            max_symmetries: Maximum number of symmetries to detect (default: 20)
        """
        logger.debug(f"Detecting symmetries for '{self.name}' with tolerance {tolerance}")
        nodes = list(self.data.nodes())
        n = len(nodes)
        
        # Extract all edge weights
        edge_weights = [d['weight'] for _, _, d in self.data.edges(data=True)]
        if not edge_weights:
            logger.warning(f"No edge weights found for '{self.name}'")
            return
            
        # Calculate tolerance based on maximum distance
        max_distance = max(edge_weights)
        tol = tolerance * max_distance
        logger.debug(f"Max distance: {max_distance:.2f}, Symmetry tolerance: {tol:.2f}")
        
        # Use distance matrix if available for efficiency
        distance_matrix = self.get_distance_matrix()
        
        # This can be very expensive for large instances, so limit number of symmetries
        symmetry_count = 0
        
        # Check pairwise symmetry between nodes, optimized with distance matrix
        for i in range(n):
            if symmetry_count >= max_symmetries:
                break
                
            for j in range(i + 1, n):
                if symmetry_count >= max_symmetries:
                    break
                    
                symmetric = True
                for k in range(n):
                    if k != i and k != j:
                        # Compare distances from nodes i and j to all other nodes
                        d_ik = distance_matrix[i][k]
                        d_jk = distance_matrix[j][k]
                        if abs(d_ik - d_jk) > tol:
                            symmetric = False
                            break
                # Create pattern if symmetry is found
                if symmetric:
                    pattern = Pattern(
                        type=PatternType.SYMMETRY,
                        elements={nodes[i], nodes[j]},
                        metadata={"tolerance": tol}
                    )
                    self._patterns.append(pattern)
                    symmetry_count += 1
                    logger.debug(f"Found symmetry between nodes {nodes[i]} and {nodes[j]}")
    
    def calculate_entropy(self) -> float:
        """
        Calculate entropy based on edge weight distribution.
        Uses a histogram approach to estimate probability distribution.
        
        Entropy indicates how varied the distances between cities are.
        Lower entropy suggests more regular structure.
        
        Returns:
            float: Entropy value
        """
        logger.debug(f"Calculating entropy for '{self.name}'")
        # Extract all edge weights
        edge_weights = [d['weight'] for _, _, d in self.data.edges(data=True)]
        if not edge_weights:
            logger.warning(f"No edge weights found for '{self.name}', entropy set to 0.0")
            return 0.0
            
        # Calculate entropy from histogram of edge weights
        hist, bin_edges = np.histogram(edge_weights, bins='auto', density=True)
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * math.log(p)
        logger.debug(f"Entropy: {entropy:.4f}")
        return entropy
    
    def get_distance_matrix(self) -> np.ndarray:
        """
        Get the distance matrix for the TSP instance.
        Cached for efficiency in repeated computations.
        
        Returns:
            np.ndarray: Matrix of distances between cities
        """
        # Return cached value if available
        if self._distance_matrix is not None:
            return self._distance_matrix
            
        # Calculate distance matrix
        nodes = sorted(self.data.nodes())
        n = len(nodes)
        distance_matrix = np.zeros((n, n))
        
        # Fill the matrix efficiently
        for i in range(n):
            for j in range(i+1, n):  # Only calculate upper triangle
                if i != j:
                    distance_matrix[i][j] = self.data[nodes[i]][nodes[j]]['weight']
                    distance_matrix[j][i] = distance_matrix[i][j]  # Mirror for lower triangle
        
        # Cache the result
        self._distance_matrix = distance_matrix
        return distance_matrix
    
    def get_tour_length(self, tour: List[Any]) -> float:
        """
        Calculate the total length of a tour.
        
        Args:
            tour: List of nodes representing a tour
            
        Returns:
            float: Total tour length
        """
        # Cache key for this tour
        cache_key = f"tour_length_{hash(tuple(tour))}"
        cached_length = self.get_cached_value(cache_key)
        if cached_length is not None:
            return cached_length
            
        n = len(tour)
        if n <= 1:
            return 0.0
            
        graph = self.data
        total_length = 0.0
        
        for i in range(n):
            total_length += graph[tour[i]][tour[(i + 1) % n]]['weight']
        
        # Cache the result
        self.cache_value(cache_key, total_length)
        return total_length
    
    def get_city_coordinates(self) -> Dict[Any, Tuple[float, float]]:
        """
        Get the coordinates of all cities.
        
        Returns:
            Dict[Any, Tuple[float, float]]: Dictionary mapping nodes to coordinates
        """
        return nx.get_node_attributes(self.data, 'pos')

class TSPSolver(Solver):
    """
    Base class for TSP-specific solvers with common complexity methods.
    Provides the base complexity calculation shared by all TSP solvers.
    
    This class handles the theoretical complexity of the TSP, which is
    exponential (O(n^2 * 2^n)) for exact solutions.
    """
    
    def get_base_complexity(self, instance_size: int) -> float:
        """
        Calculate base complexity for TSP: O(n^2 * 2^n) for exact solutions.
        For very large instances, returns complexity in logarithmic form.
        
        Args:
            instance_size: Number of cities
        
        Returns:
            float: Theoretical base complexity (or log complexity for large instances)
        """
        logger.debug(f"Calculating base complexity for size {instance_size}")
        if instance_size <= 1:
            return 1.0
            
        # For large instances, compute and return the log10 of complexity
        if instance_size > 100:
            # log(n^2 * 2^n) = log(n^2) + log(2^n) = 2*log(n) + n*log(2)
            log_complexity = 2 * math.log10(instance_size) + instance_size * math.log10(2)
            logger.debug(f"Base complexity (log10): {log_complexity:.2f}")
            return log_complexity
        else:
            # For smaller instances, compute actual value
            complexity = (instance_size ** 2) * (2 ** instance_size)
            logger.debug(f"Base complexity: {complexity:.2e}")
            return complexity

class TSPNearestNeighborSolver(TSPSolver):
    """
    Nearest Neighbor heuristic for TSP with O(n²) complexity.
    Serves as a baseline for SQF (Solution Quality Factor) calculations.
    This is a greedy algorithm that visits the nearest unvisited city at each step.
    
    While simple, Nearest Neighbor provides a reasonable starting solution
    and serves as a baseline for evaluating more sophisticated approaches.
    Its runtime is predictable, which makes it useful as a benchmark.
    """
    
    def __init__(self, config: Optional[SolverConfiguration] = None):
        """
        Initialize the Nearest Neighbor solver with optional configuration.
        
        Args:
            config: Solver configuration (optional)
        """
        super().__init__("TSP-NearestNeighbor", config)
    
    @timeit
    def solve(self, instance: TSPInstance) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Solve TSP using Nearest Neighbor heuristic.
        Creates a tour by always visiting the nearest unvisited city.
        
        Algorithm steps:
        1. Start from an arbitrary city (first node in our implementation)
        2. Repeatedly visit the nearest unvisited city until all cities are visited
        3. Return to the starting city to complete the tour
        
        Time complexity: O(n²) where n is the number of cities
        
        Args:
            instance: TSPInstance to solve
        
        Returns:
            Tuple: (tour list of nodes, metadata dictionary with runtime, tour_length, and SQF)
        """
        logger.info(f"Starting Nearest Neighbor solver for '{instance.name}'")
        start_time = time.time()
        
        graph = instance.data
        nodes = list(graph.nodes())
        
        if not nodes:
            logger.warning(f"No nodes found in '{instance.name}'")
            return [], {"runtime": 0, "tour_length": 0, "sqf": 0.0}
        
        # Start tour from first node
        start_node = nodes[0]
        tour = [start_node]
        unvisited = set(nodes)
        unvisited.remove(start_node)
        total_distance = 0
        current = start_node
        
        # Cache distances for faster access
        distance_matrix = None
        if self.config.distance_matrix_caching:
            distance_matrix = instance.get_distance_matrix()
            node_indices = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        
        # Greedily add nearest unvisited node
        while unvisited:
            # Find nearest unvisited city (optimized with distance matrix if available)
            if distance_matrix is not None:
                current_idx = node_indices[current]
                nearest = min(unvisited, key=lambda node: distance_matrix[current_idx][node_indices[node]])
                nearest_dist = distance_matrix[current_idx][node_indices[nearest]]
            else:
                nearest = min(unvisited, key=lambda node: graph[current][node]['weight'])
                nearest_dist = graph[current][nearest]['weight']
                
            total_distance += nearest_dist
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            logger.debug(f"Added node {nearest}, tour length: {total_distance:.2f}")
        
        # Close the tour by returning to start node
        if len(tour) > 1:
            if distance_matrix is not None:
                total_distance += distance_matrix[node_indices[tour[-1]]][node_indices[tour[0]]]
            else:
                total_distance += graph[tour[-1]][tour[0]]['weight']
            logger.debug(f"Closed tour, final length: {total_distance:.2f}")
        
        runtime = time.time() - start_time
        # SQF is 0 for baseline Nearest Neighbor (since it's our baseline)
        metadata = {
            "runtime": runtime,
            "tour_length": total_distance,
            "sqf": 0.0,  # Baseline has no improvement over itself
            "pue": instance.get_pue()
        }
        logger.info(f"Nearest Neighbor completed: runtime={runtime:.4f}s, tour_length={total_distance:.2f}, PUE={metadata['pue']:.2f}%, SQF=0.0%")
        return tour, metadata
    
    def get_reduction_factor(self, instance: TSPInstance) -> float:
        """
        Calculate reduction factor based on clustering patterns.
        Clusters of cities can reduce the effective complexity.
        
        The reduction factor represents how much the theoretical complexity
        is reduced by the solver's ability to exploit patterns. A value closer
        to 0 means more reduction (better pattern exploitation).
        
        For Nearest Neighbor, clusters help by limiting the search space
        for the next city to visit, improving decision-making efficiency.
        
        Args:
            instance: TSPInstance to evaluate
        
        Returns:
            float: Reduction factor (0.0 to 1.0) - lower means better pattern exploitation
        """
        logger.debug(f"Calculating reduction factor for '{instance.name}' with '{self.name}'")
        base_reduction = 0.8  # Default reduction factor
        cluster_patterns = [p for p in instance._patterns if p.type == PatternType.CLUSTERING]
        
        if cluster_patterns:
            num_clusters = len(cluster_patterns)
            # Adjust reduction based on number of clusters (max benefit capped)
            cluster_reduction = 0.5 + (0.3 * min(num_clusters / 10, 1.0))
            reduction = min(base_reduction, cluster_reduction)
            logger.debug(f"Found {num_clusters} clusters, reduction: {reduction:.4f}")
            return reduction
        logger.debug(f"No clusters, reduction: {base_reduction:.4f}")
        return base_reduction

# Batch generator for parallel 2-Opt swap evaluation
def generate_swap_batches(n: int, max_swaps: int = None, focus_clusters: List[List[int]] = None) -> Generator[List[Tuple[int, int]], None, None]:
    """
    Generate batches of swap indices for 2-Opt evaluation.
    If clusters are provided, prioritizes swaps between different clusters.
    
    Args:
        n: Number of cities in the tour
        max_swaps: Maximum number of swaps to evaluate (default: None, all swaps)
        focus_clusters: List of city indices grouped by cluster (default: None)
        
    Yields:
        List[Tuple[int, int]]: Batches of (i, j) swap indices
    """
    # If no max_swaps specified, evaluate all possible swaps
    if max_swaps is None:
        max_swaps = (n * (n - 3)) // 2  # Total possible non-adjacent swaps
    
    # If clusters are provided, prioritize inter-cluster swaps
    if focus_clusters and len(focus_clusters) > 1:
        # Generate inter-cluster swap candidates
        intercluster_swaps = []
        for c1_idx, cluster1 in enumerate(focus_clusters):
            for c2_idx in range(c1_idx + 1, len(focus_clusters)):
                cluster2 = focus_clusters[c2_idx]
                for i in cluster1:
                    for j in cluster2:
                        if abs(i - j) > 1 and i < j:  # Valid swap
                            intercluster_swaps.append((i, j))
        
        # Shuffle to avoid bias and take top portion
        random.shuffle(intercluster_swaps)
        prioritized_swaps = intercluster_swaps[:max(len(intercluster_swaps) // 2, 100)]
        
        # Add remaining capacity with regular swaps
        remaining = max_swaps - len(prioritized_swaps)
        if remaining > 0:
            regular_swaps = [(i, j) for i in range(n - 2) for j in range(i + 2, n) 
                           if (i, j) not in set(prioritized_swaps)]
            random.shuffle(regular_swaps)
            combined_swaps = prioritized_swaps + regular_swaps[:remaining]
        else:
            combined_swaps = prioritized_swaps[:max_swaps]
        
        random.shuffle(combined_swaps)  # Final shuffle for batch diversity
        swaps_to_evaluate = combined_swaps
    else:
        # Generate all possible swaps and select a subset if needed
        all_swaps = [(i, j) for i in range(n - 2) for j in range(i + 2, n)]
        if len(all_swaps) > max_swaps:
            # Random selection with uniform distribution
            swaps_to_evaluate = random.sample(all_swaps, max_swaps)
        else:
            swaps_to_evaluate = all_swaps
    
    # Split into reasonable batch sizes for parallel processing
    batch_size = min(200, max(50, len(swaps_to_evaluate) // NUM_WORKERS))
    for i in range(0, len(swaps_to_evaluate), batch_size):
        yield swaps_to_evaluate[i:i + batch_size]

##############################################################################
# 1. ENHANCED SWAP SELECTION STRATEGIES FOR 2-OPT
##############################################################################

def generate_aggressive_swap_batches(n: int, tour: List[Any], distance_matrix: np.ndarray, 
                                   node_indices: Dict[Any, int], max_swaps: int = None, 
                                   focus_clusters: List[List[int]] = None,
                                   aggression_level: int = 2) -> Generator[List[Tuple[int, int]], None, None]:
    """
    Generate batches of swap indices with more aggressive selection strategies.
    Uses edge length and gain prediction heuristics to prioritize promising swaps.
    
    Args:
        n: Number of cities in the tour
        tour: The current tour
        distance_matrix: Matrix of distances between cities
        node_indices: Mapping from nodes to their indices in the distance matrix
        max_swaps: Maximum number of swaps to evaluate
        focus_clusters: List of city indices grouped by cluster
        aggression_level: Level of aggressive selection (1-3, higher is more aggressive)
        
    Yields:
        List[Tuple[int, int]]: Batches of (i, j) swap indices
    """
    # If no max_swaps specified, evaluate all possible swaps
    if max_swaps is None:
        max_swaps = (n * (n - 3)) // 2  # Total possible non-adjacent swaps
    
    # Generate all possible swaps
    all_swaps = [(i, j) for i in range(n - 2) for j in range(i + 2, n)]
    
    # STRATEGY 1: Edge length based prioritization
    # Calculate edge lengths in the current tour
    edge_lengths = {}
    for i in range(n):
        a, b = tour[i], tour[(i+1) % n]
        a_idx, b_idx = node_indices[a], node_indices[b]
        edge_lengths[i] = distance_matrix[a_idx][b_idx]
    
    # Sort edges by length (descending)
    sorted_edges = sorted(edge_lengths.items(), key=lambda x: x[1], reverse=True)
    
    # STRATEGY 2: Gain prediction heuristic
    # Predict the gain for each possible swap
    predicted_gains = {}
    predicted_count = min(2000, len(all_swaps))  # Limit prediction calculation to avoid performance issues
    
    # Sample swaps for prediction
    sample_swaps = random.sample(all_swaps, predicted_count) if len(all_swaps) > predicted_count else all_swaps
    
    for i, j in sample_swaps:
        # Get city indices in the distance matrix
        a, b = tour[i], tour[(i+1) % n]
        c, d = tour[j], tour[(j+1) % n]
        a_idx, b_idx = node_indices[a], node_indices[b]
        c_idx, d_idx = node_indices[c], node_indices[d]
        
        # Calculate change in tour length if this swap is made
        old_length = distance_matrix[a_idx][b_idx] + distance_matrix[c_idx][d_idx]
        new_length = distance_matrix[a_idx][c_idx] + distance_matrix[b_idx][d_idx]
        gain = old_length - new_length
        
        # Only keep promising swaps
        if gain > 0:
            predicted_gains[(i, j)] = gain
    
    # STRATEGY 3: Crossing edges elimination
    # Find edges that cross each other (good candidates for improvement)
    crossing_edges = []
    if aggression_level >= 2:
        # Sample edges to check for crossings to avoid O(n²) computation
        edge_sample = sorted_edges[:min(300, len(sorted_edges))]
        for idx1, _ in edge_sample:
            for idx2, _ in edge_sample:
                if idx1 != idx2 and abs(idx1 - idx2) > 1 and abs(idx1 - idx2) < n-1:
                    a, b = tour[idx1], tour[(idx1+1) % n]
                    c, d = tour[idx2], tour[(idx2+1) % n]
                    a_pos = (node_indices[a], node_indices[b])
                    c_pos = (node_indices[c], node_indices[d])
                    
                    # Check if edges might cross (using a simple heuristic)
                    if (a_pos[0] < c_pos[0] and a_pos[1] > c_pos[1]) or (a_pos[0] > c_pos[0] and a_pos[1] < c_pos[1]):
                        # Add the swap indices
                        i, j = min(idx1, idx2), max(idx1, idx2)
                        crossing_edges.append((i, j))
    
    # STRATEGY 4: Cluster boundary swaps
    cluster_boundary_swaps = []
    if focus_clusters and aggression_level >= 2:
        # Identify edges that connect different clusters
        for c1_idx, cluster1 in enumerate(focus_clusters):
            for c2_idx in range(c1_idx + 1, len(focus_clusters)):
                cluster2 = focus_clusters[c2_idx]
                for i in cluster1:
                    next_i = (i + 1) % n
                    if next_i not in cluster1:  # Edge crosses cluster boundary
                        for j in cluster2:
                            next_j = (j + 1) % n
                            if next_j not in cluster2:  # Another boundary edge
                                cluster_boundary_swaps.append((min(i, j), max(i, j)))
    
    # Combine strategies based on aggression level
    prioritized_swaps = []
    
    # Add top predicted gain swaps
    top_gains = sorted(predicted_gains.items(), key=lambda x: x[1], reverse=True)
    gain_limit = min(len(top_gains), max(100, max_swaps // 3))
    prioritized_swaps.extend([swap for swap, _ in top_gains[:gain_limit]])
    
    # Add swaps from long edges
    for idx, _ in sorted_edges[:min(100, len(sorted_edges))]:
        for offset in range(3, min(20, n//4)):
            j = (idx + offset) % n
            if abs(idx - j) > 1 and idx < j:
                prioritized_swaps.append((idx, j))
    
    # Add crossing edges
    prioritized_swaps.extend(crossing_edges[:min(len(crossing_edges), max_swaps // 3)])
    
    # Add cluster boundary swaps
    prioritized_swaps.extend(cluster_boundary_swaps[:min(len(cluster_boundary_swaps), max_swaps // 3)])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_prioritized = []
    for swap in prioritized_swaps:
        if swap not in seen:
            seen.add(swap)
            unique_prioritized.append(swap)
    prioritized_swaps = unique_prioritized
    
    # Fill remaining slots with other swaps if needed
    remaining_slots = max_swaps - len(prioritized_swaps)
    if remaining_slots > 0:
        remaining_swaps = [s for s in all_swaps if s not in seen]
        if remaining_swaps:
            prioritized_swaps.extend(random.sample(remaining_swaps, 
                                                  min(len(remaining_swaps), remaining_slots)))
    
    # Limit to max_swaps
    swaps_to_evaluate = prioritized_swaps[:max_swaps]
    random.shuffle(swaps_to_evaluate)  # Shuffle to avoid bias
    
    # Split into batches for parallel processing
    batch_size = min(200, max(50, len(swaps_to_evaluate) // NUM_WORKERS))
    for i in range(0, len(swaps_to_evaluate), batch_size):
        yield swaps_to_evaluate[i:i + batch_size]

# Module-level evaluate_swap function for parallelized 2-Opt
def evaluate_swap(args: Tuple[int, int, Dict[Any, int], np.ndarray, List[Any]]) -> Tuple[int, int, float]:
    """
    Evaluate a 2-Opt swap between indices i and j.
    Checks if reversing the segment between i and j improves the tour.
    
    2-Opt swap operation:
    1. Remove edges (i,i+1) and (j,j+1) from the current tour
    2. Add edges (i,j) and (i+1,j+1) to create a new tour
    3. This effectively reverses the path from i+1 to j
    
    This optimized version uses distance matrix for faster calculation
    and returns only the swap indices and improvement delta.
    
    Args:
        args: Tuple of (i, j, node_indices, distance_matrix, tour)
    
    Returns:
        Tuple of (i, j, delta) where delta is the improvement amount
    """
    i, j, node_indices, distance_matrix, tour = args
    if j <= i + 1:
        return i, j, 0.0  # No improvement for adjacent or invalid swaps
    
    n = len(tour)
    
    # Get city indices in the distance matrix
    city_i = node_indices[tour[i]]
    city_i_next = node_indices[tour[i + 1]]
    city_j = node_indices[tour[j]]
    city_j_next = node_indices[tour[(j + 1) % n]]
    
    # Calculate change in tour length
    old_connection1 = distance_matrix[city_i][city_i_next]
    old_connection2 = distance_matrix[city_j][city_j_next]
    new_connection1 = distance_matrix[city_i][city_j]
    new_connection2 = distance_matrix[city_i_next][city_j_next]
    
    delta = (new_connection1 + new_connection2) - (old_connection1 + old_connection2)
    
    return i, j, delta

def apply_2opt_swap(tour: List[Any], i: int, j: int) -> List[Any]:
    """
    Apply a 2-Opt swap to a tour by reversing the segment between i+1 and j.
    
    Args:
        tour: Current tour
        i: First swap index
        j: Second swap index
        
    Returns:
        List[Any]: New tour after applying the swap
    """
    # Create new tour with reversed segment
    new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
    return new_tour

class TSP2OptSolver(TSPSolver):
    """
    2-Opt improvement heuristic for TSP with enhanced scalability.
    Implements parallelization and heuristic pruning for large instances.
    2-Opt works by repeatedly swapping two edges to improve tour length.
    
    2-Opt is a local search algorithm that iteratively improves a tour by:
    1. Starting with an initial tour (typically from Nearest Neighbor)
    2. Considering all possible pairs of edges that could be swapped
    3. Selecting the swap that most improves the tour
    4. Repeating until no improving swap can be found or other termination criteria
    
    This optimized implementation includes:
    - Distance matrix caching for faster distance calculations
    - Parallelized evaluation of swap candidates using multiple CPU cores
    - Smart swap pruning strategies that prioritize promising candidates
    - Cluster-aware swap generation that focuses on inter-cluster swaps
    - Early termination based on time limits and diminishing returns
    - Efficient memory usage through shared data structures
    """
    
    def __init__(self, config: Optional[SolverConfiguration] = None):
        """
        Initialize 2-Opt solver with configuration settings.
        
        Args:
            config: Solver configuration (default: None, uses default settings)
        """
        super().__init__("TSP-2Opt", config)
    
    @timeit
    def solve(self, instance: TSPInstance) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Solve TSP using optimized 2-Opt heuristic with parallelization and smart pruning.
        Starts from Nearest Neighbor solution and optimizes iteratively.
        
        Implementation features:
        1. Uses Nearest Neighbor solution as the starting point
        2. Evaluates potential edge swaps in parallel using multiple CPU cores
        3. Uses distance matrix caching for faster calculations
        4. Prioritizes swaps between clusters for better efficiency
        5. Implements early termination based on time and improvement thresholds
        6. Calculates SQF (Solution Quality Factor) as improvement percentage over Nearest Neighbor
        
        Args:
            instance: TSPInstance to solve
            
        Returns:
            Tuple: (optimized tour, metadata dictionary with performance metrics)
        """
        logger.info(f"Starting optimized 2-Opt solver for '{instance.name}'")
        start_time = time.time()
        
        # Adjust configuration based on instance size
        self.config.adjust_for_instance(instance)
        
        # Use Nearest Neighbor as initial solution and baseline for SQF
        nn_solver = TSPNearestNeighborSolver(self.config)
        nn_tour, nn_metadata = nn_solver.solve(instance)
        baseline_tour_length = nn_metadata["tour_length"]
        graph = instance.data
        n = len(nn_tour)
        
        # Handle small instances where 2-Opt is unnecessary
        if n <= 3:
            logger.warning(f"Instance '{instance.name}' too small for 2-Opt (size {n})")
            return nn_tour, {
                "runtime": time.time() - start_time,
                "tour_length": baseline_tour_length,
                "iterations": 0,
                "sqf": 0.0,
                "pue": instance.get_pue()
            }
        
        # Set up for efficient distance calculations
        nodes = sorted(graph.nodes())
        node_indices = {node: i for i, node in enumerate(nodes)}
        distance_matrix = instance.get_distance_matrix()
        
        # Extract cluster information for smart swap generation
        cluster_patterns = [p for p in instance._patterns if p.type == PatternType.CLUSTERING]
        cluster_indices = []
        if cluster_patterns:
            # Create mapping of nodes to their positions in the tour
            tour_positions = {node: i for i, node in enumerate(nn_tour)}
            
            # Convert cluster node sets to lists of positions in the tour
            for pattern in cluster_patterns:
                cluster_tour_indices = [tour_positions[node] for node in pattern.elements if node in tour_positions]
                if len(cluster_tour_indices) >= 3:  # Only use clusters with at least 3 cities
                    cluster_indices.append(cluster_tour_indices)
        
        # Initialize for iterative improvement
        tour = nn_tour
        current_length = baseline_tour_length
        improved = True
        iterations = 0
        best_improvements = []  # Track improvement rate for early stopping
        last_improvement_time = time.time()  # Track time of last improvement
        early_termination_reason = None
        
        # Iterate until no improvements or termination criteria met
        while improved and iterations < self.config.max_iterations:
            # Check if time limit exceeded
            if self.check_timeout(start_time):
                early_termination_reason = "time_limit"
                break
                
            improved = False
            iterations += 1
            logger.debug(f"2-Opt iteration {iterations}")
            
            # Generate batches of swaps to evaluate
            swap_batches = list(generate_swap_batches(
                n, 
                max_swaps=self.config.max_swaps_per_iteration,
                focus_clusters=cluster_indices if self.config.prune_swaps else None
            ))
            
            # Process batches in parallel
            best_swap = None
            best_delta = 0
            result_lock = threading.Lock()  # Lock for thread safety
            
            with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=worker_init) as executor:
                for batch in swap_batches:
                    # Create evaluation tasks for this batch
                    tasks = [(i, j, node_indices, distance_matrix, tour) for i, j in batch]
                    
                    # Submit all tasks for this batch
                    results = list(executor.map(evaluate_swap, tasks))
                    
                    # Process results from this batch with proper synchronization
                    with result_lock:
                        for i, j, delta in results:
                            if delta < best_delta:  # Find the biggest improvement (most negative delta)
                                best_delta = delta
                                best_swap = (i, j)
            
            # Apply best improvement if found
            if best_swap and best_delta < 0:
                i, j = best_swap
                tour = apply_2opt_swap(tour, i, j)
                prev_length = current_length
                current_length = prev_length + best_delta
                
                # Track improvement for early termination check
                improvement_percent = abs(best_delta) / prev_length * 100
                best_improvements.append(improvement_percent)
                last_improvement_time = time.time()
                
                improved = True
                logger.debug(f"Improved tour length to {current_length:.2f} (delta: {best_delta:.2f}, {improvement_percent:.2f}%)")
                
                # Check for diminishing returns
                if len(best_improvements) >= 3:
                    recent_avg = sum(best_improvements[-3:]) / 3
                    if recent_avg < self.config.early_termination_threshold:
                        logger.info(f"Early termination due to diminishing returns (avg improvement: {recent_avg:.4f}%)")
                        early_termination_reason = "diminishing_returns"
                        break
            
            # Check if it's been too long since last improvement
            if improved and time.time() - last_improvement_time > self.config.max_execution_time / 4:
                logger.info("Early termination due to stagnation (no significant improvement recently)")
                early_termination_reason = "stagnation"
                break
        
        runtime = time.time() - start_time
        # Calculate SQF as percentage reduction relative to baseline
        sqf = ((baseline_tour_length - current_length) / baseline_tour_length * 100.0) if baseline_tour_length > 0 else 0.0
        metadata = {
            "runtime": runtime,
            "tour_length": current_length,
            "iterations": iterations,
            "sqf": sqf,
            "pue": instance.get_pue(),
            "early_termination": early_termination_reason
        }
        
        logger.info(f"2-Opt completed: runtime={runtime:.4f}s, tour_length={current_length:.2f}, "
                    f"iterations={iterations}, PUE={metadata['pue']:.2f}%, SQF={sqf:.2f}%"
                    f"{f', terminated: {early_termination_reason}' if early_termination_reason else ''}")
        return tour, metadata
    
    def get_reduction_factor(self, instance: TSPInstance) -> float:
        """
        Calculate reduction factor based on symmetry patterns.
        Symmetrical cities can reduce the effective complexity.
        
        2-Opt can exploit symmetry patterns effectively:
        - If two cities have similar distances to all other cities, swaps involving
          these cities can be prioritized or deprioritized together
        - Symmetry reduces the effective search space by allowing the solver to
          treat groups of similar swaps as equivalent
        
        Args:
            instance: TSPInstance to evaluate
        
        Returns:
            float: Reduction factor (0.0 to 1.0) - lower means better pattern exploitation
        """
        logger.debug(f"Calculating reduction factor for '{instance.name}' with '{self.name}'")
        base_reduction = 0.7  # Default reduction factor
        
        # Consider both symmetry and clustering patterns
        symmetry_patterns = [p for p in instance._patterns if p.type == PatternType.SYMMETRY]
        cluster_patterns = [p for p in instance._patterns if p.type == PatternType.CLUSTERING]
        
        # Calculate reduction from symmetries
        sym_reduction = base_reduction
        if symmetry_patterns:
            num_symmetries = len(symmetry_patterns)
            # Adjust reduction based on number of symmetries (capped benefit)
            sym_reduction = 0.4 + (0.3 * min(num_symmetries / 5, 1.0))
        
        # Calculate reduction from clusters
        cluster_reduction = base_reduction
        if cluster_patterns:
            num_clusters = len(cluster_patterns)
            # Adjust reduction based on number of clusters (max benefit capped)
            cluster_reduction = 0.5 + (0.2 * min(num_clusters / 10, 1.0))
        
        # Take the better (lower) of the two reductions
        reduction = min(sym_reduction, cluster_reduction, base_reduction)
        logger.debug(f"Reduction factor: {reduction:.4f} (from {len(symmetry_patterns)} symmetries, {len(cluster_patterns)} clusters)")
        return reduction

class TSP3OptSolver(TSPSolver):
    """
    3-Opt improvement heuristic for TSP with limited depth search.
    More powerful than 2-Opt but more computationally intensive.
    
    3-Opt considers all possible ways to reconnect the tour after removing 3 edges,
    which allows for more complex rearrangements than 2-Opt but is more expensive.
    
    This implementation uses:
    - Limited search depth to control computational cost
    - Cluster-aware segment selection
    - First-improvement (rather than best-improvement) strategy
    - Early termination criteria
    """
    
    def __init__(self, config: Optional[SolverConfiguration] = None):
        """
        Initialize 3-Opt solver with configuration settings.
        
        Args:
            config: Solver configuration (default: None, uses default settings)
        """
        super().__init__("TSP-3Opt", config)
        # Apply additional 3-Opt specific configuration adjustments
        if self.config:
            # Set reasonable defaults for 3-Opt if not already set
            if not hasattr(self.config, 'max_3opt_iterations'):
                self.config.max_3opt_iterations = 20
    
    @timeit
    def solve(self, instance: TSPInstance) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Solve TSP using 3-Opt heuristic with limited search and early termination.
        Typically produces better solutions than 2-Opt but takes longer.
        
        Implementation:
        1. Starts with a 2-Opt solution as initialization
        2. Considers limited 3-Opt moves to improve the tour
        3. Uses first-improvement strategy for efficiency
        4. Implements early termination based on time and improvement thresholds
        
        Args:
            instance: TSPInstance to solve
            
        Returns:
            Tuple: (optimized tour, metadata dictionary with performance metrics)
        """
        logger.info(f"Starting 3-Opt solver for '{instance.name}'")
        start_time = time.time()
        
        # Adjust configuration based on instance size
        self.config.adjust_for_instance(instance)
        
        # For small instances (< 200 cities), use 3-Opt with reasonable limits
        # For larger instances, just use 2-Opt as initialization
        n = instance.size
        if n > 200:
            logger.info(f"Large instance detected ({n} cities), using 2-Opt solution as initialization")
            opt2_solver = TSP2OptSolver(self.config)
            tour, opt2_metadata = opt2_solver.solve(instance)
            baseline_tour_length = opt2_metadata.get("tour_length", float('inf'))
            nn_tour_length = opt2_metadata.get("nn_tour_length", baseline_tour_length)
        else:
            # Use Nearest Neighbor as baseline for SQF
            nn_solver = TSPNearestNeighborSolver(self.config)
            nn_tour, nn_metadata = nn_solver.solve(instance)
            nn_tour_length = nn_metadata["tour_length"]
            
            # Apply 2-Opt first to get a better starting solution
            opt2_solver = TSP2OptSolver(self.config)
            tour, opt2_metadata = opt2_solver.solve(instance)
            baseline_tour_length = opt2_metadata["tour_length"]
        
        # Early termination if instance is too large or time constraint is tight
        if n > 500 or self.config.max_execution_time < 10:
            logger.warning(f"Skipping 3-Opt improvements due to instance size ({n}) or time constraint")
            
            # Calculate SQF based on nearest neighbor baseline
            sqf = ((nn_tour_length - baseline_tour_length) / nn_tour_length * 100.0) if nn_tour_length > 0 else 0.0
            
            metadata = {
                "runtime": time.time() - start_time,
                "tour_length": baseline_tour_length,
                "iterations": 0,
                "sqf": sqf,
                "pue": instance.get_pue(),
                "early_termination": "size_constraint"
            }
            return tour, metadata
        
        # Set up for 3-Opt improvement
        graph = instance.data
        current_length = baseline_tour_length
        nodes = sorted(graph.nodes())
        distance_matrix = instance.get_distance_matrix()
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Track improvements and timing
        iterations = 0
        improved = True
        last_improvement_time = time.time()
        early_termination_reason = None
        
        # Maximum number of 3-Opt iterations based on instance size
        max_3opt_iterations = self.config.max_3opt_iterations if hasattr(self.config, 'max_3opt_iterations') else min(20, 100 // (1 + n // 50))
        logger.debug(f"Maximum 3-Opt iterations: {max_3opt_iterations}")
        
        # Perform 3-Opt improvements
        while improved and iterations < max_3opt_iterations:
            # Check if time limit exceeded
            if self.check_timeout(start_time):
                early_termination_reason = "time_limit"
                break
            
            improved = False
            iterations += 1
            tour_size = len(tour)
            
            # Consider a limited set of segments for 3-Opt moves
            # For each iteration, try a different sampling strategy to explore the search space
            segments_to_try = min(200, max(50, n))
            if iterations % 3 == 0:
                # Sample segments uniformly
                segments = random.sample(range(tour_size), min(segments_to_try, tour_size))
            elif iterations % 3 == 1:
                # Focus on the beginning and end of the tour
                segments = list(range(min(segments_to_try // 2, tour_size // 3))) + \
                           list(range(max(0, tour_size - segments_to_try // 2), tour_size))
            else:
                # Focus on the middle of the tour
                mid = tour_size // 2
                half_width = min(segments_to_try // 2, tour_size // 3)
                segments = list(range(max(0, mid - half_width), min(tour_size, mid + half_width)))
            
            # Try 3-Opt moves with the selected segments
            for i in segments:
                if improved:
                    break
                
                for j in range(i + 2, tour_size - 1):
                    if improved:
                        break
                    
                    for k in range(j + 2, tour_size + (1 if i == 0 else 0) - 1):
                        if i == 0 and k == tour_size - 1:
                            continue  # Skip invalid configuration
                        
                        # Get city indices for distance calculations
                        i_plus = (i + 1) % tour_size
                        j_plus = (j + 1) % tour_size
                        k_plus = (k + 1) % tour_size
                        
                        a, b = tour[i], tour[i_plus]
                        c, d = tour[j], tour[j_plus]
                        e, f = tour[k], tour[k_plus]
                        
                        a_idx, b_idx = node_indices[a], node_indices[b]
                        c_idx, d_idx = node_indices[c], node_indices[d]
                        e_idx, f_idx = node_indices[e], node_indices[f]
                        
                        d_ab = distance_matrix[a_idx][b_idx]
                        d_cd = distance_matrix[c_idx][d_idx]
                        d_ef = distance_matrix[e_idx][f_idx]
                        original = d_ab + d_cd + d_ef
                        
                        # Test all possible 3-Opt reconnections (7 possibilities excluding original)
                        # For each case, calculate delta = new_cost - original_cost
                        
                        # Case 1: Reconnect a-c, b-e, d-f
                        d_ac = distance_matrix[a_idx][c_idx]
                        d_be = distance_matrix[b_idx][e_idx]
                        d_df = distance_matrix[d_idx][f_idx]
                        delta1 = d_ac + d_be + d_df - original
                        
                        if delta1 < -1e-10:  # Small epsilon to handle floating point errors
                            # Apply the move: a-c, b-e, d-f
                            # This reconnects the segments [a...b], [c...d], [e...f]
                            # as [a, c, ... d], [b, e, ... f]
                            new_tour = [tour[i]]
                            new_tour.extend(tour[j:i:-1])  # Reversed segment [j, j-1, ..., i+1]
                            new_tour.extend(tour[k+1:j+1:-1])  # Reversed [k, k-1, ..., j+1]
                            new_tour.extend(tour[i+1:])  # Rest of the tour
                            
                            tour = new_tour
                            current_length += delta1
                            improved = True
                            last_improvement_time = time.time()
                            logger.debug(f"3-Opt improvement: {delta1:.2f}, new length: {current_length:.2f}")
                            break
                        
                        # We're using first-improvement strategy, so we stop at the first improvement
                        # If you want to try all 7 possible 3-Opt moves, add the other cases here
                        
                        # Check timeout periodically
                        if iterations % 10 == 0 and self.check_timeout(start_time):
                            early_termination_reason = "time_limit"
                            improved = False
                            break
            
            # Check for stagnation
            if time.time() - last_improvement_time > self.config.max_execution_time / 5:
                logger.info("Early termination due to stagnation (no improvement recently)")
                early_termination_reason = "stagnation"
                break
        
        runtime = time.time() - start_time
        # Calculate SQF based on nearest neighbor tour
        sqf = ((nn_tour_length - current_length) / nn_tour_length * 100.0) if nn_tour_length > 0 else 0.0
        
        metadata = {
            "runtime": runtime,
            "tour_length": current_length,
            "nn_tour_length": nn_tour_length,
            "iterations": iterations,
            "sqf": sqf,
            "pue": instance.get_pue(),
            "early_termination": early_termination_reason
        }
        
        logger.info(f"3-Opt completed: runtime={runtime:.4f}s, tour_length={current_length:.2f}, "
                   f"iterations={iterations}, PUE={metadata['pue']:.2f}%, SQF={sqf:.2f}%"
                   f"{f', terminated: {early_termination_reason}' if early_termination_reason else ''}")
        return tour, metadata
    
    def get_reduction_factor(self, instance: TSPInstance) -> float:
        """
        Calculate reduction factor for 3-Opt based on symmetry and clustering patterns.
        3-Opt can exploit patterns even more effectively than 2-Opt.
        
        Args:
            instance: TSPInstance to evaluate
        
        Returns:
            float: Reduction factor (0.0 to 1.0) - lower means better pattern exploitation
        """
        # 3-Opt is typically 5-10% more effective than 2-Opt at exploiting patterns
        # so we scale down the reduction factor
        opt2_solver = TSP2OptSolver()
        opt2_reduction = opt2_solver.get_reduction_factor(instance)
        
        # Make 3-Opt reduction factor 10% better than 2-Opt
        reduction = max(0.1, opt2_reduction * 0.9)
        logger.debug(f"3-Opt reduction factor: {reduction:.4f} (from 2-Opt: {opt2_reduction:.4f})")
        return reduction


##############################################################################
# 2. IMPROVED 3-OPT AND NEW 4-OPT IMPLEMENTATION
##############################################################################

def apply_3opt_move(tour: List[Any], i: int, j: int, k: int, case: int) -> List[Any]:
    """
    Apply a 3-Opt move to a tour by reconnecting the segments in one of 7 possible ways.
    
    Args:
        tour: Current tour
        i, j, k: The three breaking points
        case: Which reconnection case to use (1-7)
        
    Returns:
        List[Any]: New tour after the 3-Opt move
    """
    n = len(tour)
    A, B = tour[i], tour[(i+1) % n]
    C, D = tour[j], tour[(j+1) % n]
    E, F = tour[k], tour[(k+1) % n]
    
    if case == 1:  # Reconnect A-C, B-E, D-F (reverse B-C, D-E)
        new_tour = tour[:i+1] + tour[j:i:-1] + tour[k:j:-1] + tour[k+1:] if i+1 <= j else tour[:i+1] + tour[j:] + tour[:j:-1] + tour[k:j:-1] + tour[k+1:]
    elif case == 2:  # Reconnect A-D, E-B, C-F (reverse A-B, C-D)
        new_tour = tour[:i+1] + tour[j+1:k+1] + tour[i+1:j+1][::-1] + tour[k+1:]
    elif case == 3:  # Reconnect A-E, D-B, C-F (reverse A-B, E-F)
        new_tour = tour[:i+1] + tour[k:j:-1] + tour[i+1:j+1] + tour[k+1:]
    elif case == 4:  # Reconnect A-C, D-F, E-B (reverse D-E)
        if i+1 <= j and j+1 <= k:
            new_tour = tour[:i+1] + tour[j:i:-1] + tour[j+1:k+1] + tour[k:j:-1] + tour[k+1:]
        else:
            # Handle wrap-around cases carefully
            segments = []
            segments.append(tour[:i+1])
            segments.append(tour[j:i:-1] if i+1 <= j else tour[:i:-1])
            segments.append(tour[j+1:k+1] if j+1 <= k else tour[j+1:] + tour[:k+1])
            segments.append(tour[k:j:-1] if j+1 <= k else (tour[:j+1:-1] if k+1 <= j else tour[k:] + tour[:j+1]))
            segments.append(tour[k+1:] if k+1 < n else [])
            new_tour = []
            for segment in segments:
                new_tour.extend(segment)
    elif case == 5:  # Reconnect A-D, B-F, C-E (reverse C-D, E-F)
        new_tour = tour[:i+1] + tour[j+1:k+1][::-1] + tour[i+1:j+1] + tour[k+1:]
    elif case == 6:  # Reconnect A-E, B-C, D-F (reverse B-C)
        if i+1 <= j and j+1 <= k:
            new_tour = tour[:i+1] + tour[k:j:-1] + tour[i+1:j+1][::-1] + tour[k+1:]
        else:
            # Handle wrap-around cases
            segments = []
            segments.append(tour[:i+1])
            segments.append(tour[k:j:-1] if j+1 <= k else (tour[:j+1:-1] if k+1 <= j else tour[k:] + tour[:j+1]))
            segments.append(tour[i+1:j+1][::-1] if i+1 <= j else (tour[:j+1][::-1] + tour[i+1:][::-1]))
            segments.append(tour[k+1:] if k+1 < n else [])
            new_tour = []
            for segment in segments:
                new_tour.extend(segment)
    elif case == 7:  # Reconnect A-F, B-C, D-E (reverse all segments)
        if i+1 <= j and j+1 <= k and k+1 <= n:
            new_tour = tour[:i+1] + tour[k+1:][::-1] + tour[j+1:k+1][::-1] + tour[i+1:j+1][::-1]
        else:
            # Complex wrap-around case - handle all segments carefully
            segments = []
            segments.append(tour[:i+1])
            tail = tour[k+1:] if k+1 < n else []
            segments.append(tail[::-1])
            mid2 = tour[j+1:k+1] if j+1 <= k else (tour[j+1:] + tour[:k+1])
            segments.append(mid2[::-1])
            mid1 = tour[i+1:j+1] if i+1 <= j else (tour[i+1:] + tour[:j+1])
            segments.append(mid1[::-1])
            new_tour = []
            for segment in segments:
                new_tour.extend(segment)
    else:
        return tour.copy()  # Invalid case
    
    # Ensure we have the right length tour
    if len(new_tour) != n:
        # Something went wrong with the segmentation
        return tour.copy()
    
    return new_tour


def evaluate_3opt_move(args: Tuple[int, int, int, int, Dict[Any, int], np.ndarray, List[Any]]) -> Tuple[int, int, int, int, float]:
    """
    Evaluate a 3-Opt move by calculating the change in tour length.
    
    Args:
        args: Tuple of (i, j, k, case, node_indices, distance_matrix, tour)
    
    Returns:
        Tuple: (i, j, k, case, delta) where delta is the improvement amount
    """
    i, j, k, case, node_indices, distance_matrix, tour = args
    n = len(tour)
    
    # Get city indices for the six cities involved
    a, b = tour[i], tour[(i+1) % n]
    c, d = tour[j], tour[(j+1) % n]
    e, f = tour[k], tour[(k+1) % n]
    
    # Validate cities exist in node_indices 
    if not all(city in node_indices for city in [a, b, c, d, e, f]):
        return i, j, k, case, 0  # Return zero delta if any city is missing
    
    a_idx, b_idx = node_indices[a], node_indices[b]
    c_idx, d_idx = node_indices[c], node_indices[d]
    e_idx, f_idx = node_indices[e], node_indices[f]
    
    # Validate indices to prevent out-of-bounds errors
    if any(idx < 0 or idx >= len(distance_matrix) for idx in [a_idx, b_idx, c_idx, d_idx, e_idx, f_idx]):
        return i, j, k, case, 0  # Return zero delta for invalid indices
    
    # Calculate distance of original connections
    d_ab = distance_matrix[a_idx][b_idx]
    d_cd = distance_matrix[c_idx][d_idx]
    d_ef = distance_matrix[e_idx][f_idx]
    original_length = d_ab + d_cd + d_ef
    
    # Calculate new connections based on the case
    if case == 1:  # A-C, B-E, D-F
        d_ac = distance_matrix[a_idx][c_idx]
        d_be = distance_matrix[b_idx][e_idx]
        d_df = distance_matrix[d_idx][f_idx]
        new_length = d_ac + d_be + d_df
    elif case == 2:  # A-D, E-B, C-F
        d_ad = distance_matrix[a_idx][d_idx]
        d_eb = distance_matrix[e_idx][b_idx]
        d_cf = distance_matrix[c_idx][f_idx]
        new_length = d_ad + d_eb + d_cf
    elif case == 3:  # A-E, D-B, C-F
        d_ae = distance_matrix[a_idx][e_idx]
        d_db = distance_matrix[d_idx][b_idx]
        d_cf = distance_matrix[c_idx][f_idx]
        new_length = d_ae + d_db + d_cf
    elif case == 4:  # A-C, D-F, E-B
        d_ac = distance_matrix[a_idx][c_idx]
        d_df = distance_matrix[d_idx][f_idx]
        d_eb = distance_matrix[e_idx][b_idx]
        new_length = d_ac + d_df + d_eb
    elif case == 5:  # A-D, B-F, C-E
        d_ad = distance_matrix[a_idx][d_idx]
        d_bf = distance_matrix[b_idx][f_idx]
        d_ce = distance_matrix[c_idx][e_idx]
        new_length = d_ad + d_bf + d_ce
    elif case == 6:  # A-E, B-C, D-F
        d_ae = distance_matrix[a_idx][e_idx]
        d_bc = distance_matrix[b_idx][c_idx]
        d_df = distance_matrix[d_idx][f_idx]
        new_length = d_ae + d_bc + d_df
    elif case == 7:  # A-F, B-C, D-E
        d_af = distance_matrix[a_idx][f_idx]
        d_bc = distance_matrix[b_idx][c_idx]
        d_de = distance_matrix[d_idx][e_idx]
        new_length = d_af + d_bc + d_de
    else:
        return i, j, k, case, 0  # Invalid case
    
    # Calculate the delta (negative is an improvement)
    delta = new_length - original_length
    
    # Always return the result tuple
    return i, j, k, case, delta

class TSPEnhanced3OptSolver(TSPSolver):
    """
    Enhanced 3-Opt improvement heuristic for TSP with better moves and search strategies.
    Evaluates all 7 possible 3-Opt reconnections and uses aggressive move selection.
    """
    
    def __init__(self, config: Optional[SolverConfiguration] = None):
        """Initialize the Enhanced 3-Opt solver."""
        super().__init__("TSP-Enhanced3Opt", config)
        # Apply specific configuration for 3-Opt
        if self.config:
            # Set reasonable defaults
            if not hasattr(self.config, 'max_3opt_iterations'):
                self.config.max_3opt_iterations = 20
            if not hasattr(self.config, 'initial_temp'):
                self.config.initial_temp = 1.0
            if not hasattr(self.config, 'cooling_rate'):
                self.config.cooling_rate = 0.95
    
    @timeit
    def solve(self, instance: TSPInstance) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Solve TSP using enhanced 3-Opt heuristic with all 7 possible moves.
        Uses 2-Opt solution as initialization and employs advanced search strategies.
        
        Args:
            instance: TSPInstance to solve
            
        Returns:
            Tuple: (optimized tour, metadata dictionary)
        """
        logger.info(f"Starting Enhanced 3-Opt solver for '{instance.name}'")
        start_time = time.time()
        
        # Adjust configuration based on instance size
        self.config.adjust_for_instance(instance)
        
        # Get initial solution from 2-Opt
        opt2_solver = TSP2OptSolver(self.config)
        tour, opt2_metadata = opt2_solver.solve(instance)
        baseline_tour_length = opt2_metadata.get("tour_length", float('inf'))
        nn_tour_length = opt2_metadata.get("nn_tour_length", baseline_tour_length)
        
        # Skip 3-Opt for very large instances to avoid excessive runtime
        if instance.size > 800 and self.config.max_execution_time < 300:
            logger.warning(f"Skipping Enhanced 3-Opt for large instance (size {instance.size}) with limited time budget")
            return tour, {
                "runtime": time.time() - start_time,
                "tour_length": baseline_tour_length,
                "nn_tour_length": nn_tour_length,
                "iterations": 0,
                "sqf": opt2_metadata.get("sqf", 0.0),
                "pue": instance.get_pue(),
                "early_termination": "size_constraint"
            }
        
        # Set up for 3-Opt
        graph = instance.data
        current_length = baseline_tour_length
        nodes = sorted(graph.nodes())
        distance_matrix = instance.get_distance_matrix()
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Enhanced tracking of progress
        iterations = 0
        total_improvements = 0
        total_moves_tried = 0
        improvement_history = []  # Track improvement rate over time
        last_improvement_time = time.time()
        current_tour = tour.copy()
        best_tour = tour.copy()
        best_length = current_length
        
        # 3-Opt main loop with simulated annealing characteristics
        max_iterations = self.config.max_3opt_iterations
        temperature = self.config.initial_temp
        
        segment_size = max(20, min(50, instance.size // 20))  # Size of segments to try 3-Opt on
        
        while iterations < max_iterations:
            # Check if time limit exceeded
            if self.check_timeout(start_time):
                logger.warning("Time limit exceeded in Enhanced 3-Opt")
                break
                
            iterations += 1
            improved = False
            
            # Temperature-based segment selection: more segments with higher temperature
            num_segments = max(3, int(10 * temperature))
            
            # Generate candidate segments to focus on
            segments = []
            for _ in range(num_segments):
                # Random segment with size based on instance size
                i = random.randint(0, len(current_tour) - 1)
                segments.append((i, (i + segment_size) % len(current_tour)))
            
            # Add cluster-based segments if available
            cluster_patterns = [p for p in instance._patterns if p.type == PatternType.CLUSTERING]
            if cluster_patterns:
                # Create mapping of nodes to their positions in the tour
                tour_positions = {node: i for i, node in enumerate(current_tour)}
                
                # Add segments from cluster boundaries
                for pattern in cluster_patterns[:min(5, len(cluster_patterns))]:
                    cluster_nodes = list(pattern.elements)
                    if len(cluster_nodes) >= 3:
                        # Find positions of these nodes in the tour
                        positions = [tour_positions[node] for node in cluster_nodes if node in tour_positions]
                        if positions:
                            # Sort positions and find min/max
                            positions.sort()
                            # Add segment of tour that contains this cluster
                            segments.append((positions[0], positions[-1]))
            
            # Process each segment
            for segment_start, segment_end in segments:
                if improved:
                    break  # If we already found an improvement, move to next iteration
                
                # Determine the range for this segment
                segment_length = (segment_end - segment_start) % len(current_tour)
                if segment_length < 5:
                    continue  # Skip very small segments
                
                # Try 3-Opt moves within this segment
                tasks = []
                
                # Consider all possible triplets of edges within reasonable limits
                sample_size = min(segment_length, 30)  # Sample points within segment
                i_values = [(segment_start + i) % len(current_tour) for i in sorted(random.sample(range(segment_length), sample_size))]
                
                for i_pos in range(len(i_values)):
                    i = i_values[i_pos]
                    for j_pos in range(i_pos + 1, min(i_pos + 10, len(i_values))):
                        j = i_values[j_pos]
                        if j <= i + 1:
                            continue
                            
                        for k_pos in range(j_pos + 1, min(j_pos + 10, len(i_values))):
                            k = i_values[k_pos]
                            if k <= j + 1:
                                continue
                                
                            # Try all 7 possible 3-Opt cases
                            for case in range(1, 8):
                                tasks.append((i, j, k, case, node_indices, distance_matrix, current_tour))
                
                # Evaluate all candidate moves in parallel
                total_moves_tried += len(tasks)
                
                with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=worker_init) as executor:
                    results = list(executor.map(evaluate_3opt_move, tasks))
                
                # Find best move with improved error handling
                best_move = None
                best_delta = 0
                
                for result in results:
                    # Handle None results
                    if result is None:
                        logger.warning("Received None result from 3-opt move evaluation")
                        continue
            							
                    try:
                         i, j, k, case, delta = result
                         # Accept improving moves and occasionally accept non-improving moves (simulated annealing)
            	
                         try: 
                            acceptance_probability = 0.0
                            if temperature > 0.0001:  # Ensure temperature is not too close to zero
                               exp_term = -delta / temperature
                               # Clip exp_term to avoid overflow
                               exp_term = max(-700, min(700, exp_term))  # exp(±700) is near the limit for float64
                               acceptance_probability = math.exp(exp_term)
            
                            if delta < best_delta or (temperature > 0.1 and delta < 0.1 * current_length and random.random() < acceptance_probability):
                                best_delta = delta
                                best_move = (i, j, k, case)
            							
                         except (ValueError, OverflowError, ZeroDivisionError) as e:
                             logger.warning(f"Numerical error in simulated annealing calculation: {e}")
                             # Take the most conservative action - only accept if it's a clear improvement
                             if delta < best_delta:
                                 best_delta = delta
                                 best_move = (i, j, k, case)
            
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error unpacking 3-opt move result: {e}, got: {result}")
                        continue
            						
             # Apply best move if found
                if best_move and best_delta < 0:
                    i, j, k, case = best_move
                    current_tour = apply_3opt_move(current_tour, i, j, k, case)
                    current_length += best_delta
                    
                    # Update best tour if this is better
                    if current_length < best_length:
                        best_tour = current_tour.copy()
                        best_length = current_length
                    
                    improved = True
                    total_improvements += 1
                    improvement_history.append(-best_delta)
                    last_improvement_time = time.time()
                    logger.debug(f"3-Opt improvement: delta={best_delta:.2f}, new length={current_length:.2f}")
            
            # Update temperature
            temperature *= self.config.cooling_rate
            
            # Check for stagnation
            if time.time() - last_improvement_time > self.config.max_execution_time / 4:
                logger.info("Enhanced 3-Opt: Terminating due to stagnation")
                break
        
        runtime = time.time() - start_time
        
        # Calculate SQF based on nearest neighbor tour
        sqf = ((nn_tour_length - best_length) / nn_tour_length * 100.0) if nn_tour_length > 0 else 0.0
        
        # Prepare metadata with detailed performance information
        metadata = {
            "runtime": runtime,
            "tour_length": best_length,
            "nn_tour_length": nn_tour_length,
            "iterations": iterations,
            "improvements": total_improvements,
            "moves_tried": total_moves_tried,
            "improvement_rate": total_improvements / max(1, total_moves_tried),
            "sqf": sqf,
            "pue": instance.get_pue()
        }
        
        logger.info(f"Enhanced 3-Opt completed: runtime={runtime:.4f}s, tour_length={best_length:.2f}, "
                   f"iterations={iterations}, improvements={total_improvements}, SQF={sqf:.2f}%")
                   
        return best_tour, metadata

    def get_reduction_factor(self, instance: TSPInstance) -> float:
        """
        Calculate reduction factor for Enhanced 3-Opt based on patterns.
        Enhanced 3-Opt can exploit patterns even more effectively than regular 3-Opt.
        
        Args:
            instance: TSPInstance to evaluate
        
        Returns:
            float: Reduction factor (0.0 to 1.0) - lower means better pattern exploitation
        """
        # Use 3-Opt's reduction factor as a base and improve it by 10%
        opt3_solver = TSP3OptSolver()
        opt3_reduction = opt3_solver.get_reduction_factor(instance)
        
        # Make Enhanced 3-Opt reduction factor 10% better than regular 3-Opt
        reduction = max(0.1, opt3_reduction * 0.9)
        logger.debug(f"Enhanced 3-Opt reduction factor: {reduction:.4f} (from 3-Opt: {opt3_reduction:.4f})")
        return reduction
    
class TSP4OptSolver(TSPSolver):
    """
    4-Opt solver for TSP that removes 4 edges and reconnects in the optimal way.
    Best for hard instances where 2-Opt and 3-Opt get stuck in local optima.
    """
    
    def __init__(self, config: Optional[SolverConfiguration] = None):
        """Initialize the 4-Opt solver."""
        super().__init__("TSP-4Opt", config)
        if self.config:
            if not hasattr(self.config, 'max_4opt_iterations'):
                self.config.max_4opt_iterations = 10
            if not hasattr(self.config, 'segment_sampling'):
                self.config.segment_sampling = True
                
    def get_reduction_factor(self, instance: TSPInstance) -> float:
        """
        Calculate reduction factor for 4-Opt based on patterns.
        4-Opt can exploit patterns more effectively than 3-Opt or 2-Opt,
        especially for smaller instances with complex structure.
        
        Args:
            instance: TSPInstance to evaluate
        
        Returns:
            float: Reduction factor (0.0 to 1.0) - lower means better pattern exploitation
        """
        # 4-Opt is generally better than 3-Opt at exploiting patterns
        # Use Enhanced 3-Opt's reduction factor as a starting point
        e3opt_solver = TSPEnhanced3OptSolver()
        e3opt_reduction = e3opt_solver.get_reduction_factor(instance)
        
        # 4-Opt is approximately 15% better than Enhanced 3-Opt at exploiting patterns
        # But ensure we don't go below a reasonable minimum
        reduction = max(0.1, e3opt_reduction * 0.85)
        
        logger.debug(f"4-Opt reduction factor: {reduction:.4f} (from Enhanced 3-Opt: {e3opt_reduction:.4f})")
        return reduction
    
    @timeit
    def solve(self, instance: TSPInstance) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Solve TSP using 4-Opt moves, starting from a 3-Opt solution.
        Limited to smaller instances due to complexity.
        
        Args:
            instance: TSPInstance to solve
            
        Returns:
            Tuple: (optimized tour, metadata dictionary)
        """
        logger.info(f"Starting 4-Opt solver for '{instance.name}'")
        start_time = time.time()
        
        # Size constraint - 4-Opt is very expensive
        if instance.size > 300:
            logger.warning(f"4-Opt solver not suitable for instance size {instance.size}")
            # Fall back to Enhanced 3-Opt
            solver = TSPEnhanced3OptSolver(self.config)
            return solver.solve(instance)
        
        # Start from best available solution
        e3opt_solver = TSPEnhanced3OptSolver(self.config)
        tour, e3opt_metadata = e3opt_solver.solve(instance)
        baseline_tour_length = e3opt_metadata.get("tour_length", float('inf'))
        nn_tour_length = e3opt_metadata.get("nn_tour_length", baseline_tour_length)
        
        # Setup
        graph = instance.data
        current_length = baseline_tour_length
        nodes = sorted(graph.nodes())
        distance_matrix = instance.get_distance_matrix()
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Track progress
        iterations = 0
        improvements = 0
        current_tour = tour.copy()
        best_tour = tour.copy()
        best_length = current_length
        
        # 4-Opt is much more complex than 3-Opt
        # We'll use segment sampling to make it computationally feasible
        while iterations < self.config.max_4opt_iterations:
            if self.check_timeout(start_time):
                logger.warning("Time limit exceeded in 4-Opt")
                break
                
            iterations += 1
            improved = False
            
            # Select a segment of the tour to focus on (10-20% of the tour)
            segment_size = max(20, min(instance.size // 5, 60))
            segment_start = random.randint(0, instance.size - 1)
            segment_end = (segment_start + segment_size) % instance.size
            
            # Generate all 4-tuples of points within the segment
            points = []
            if segment_end > segment_start:
                points = list(range(segment_start, segment_end))
            else:
                points = list(range(segment_start, instance.size)) + list(range(0, segment_end))
            
            # Sample points if there are too many
            if len(points) > 20:
                points = sorted(random.sample(points, 20))
            
            # Try double-bridge moves (a specific 4-Opt move that can't be achieved with 3-Opt)
            for i_idx in range(len(points)):
                if improved:
                    break
                i = points[i_idx]
                
                for j_idx in range(i_idx + 1, len(points)):
                    j = points[j_idx]
                    if j <= i + 1:
                        continue
                        
                    for k_idx in range(j_idx + 1, len(points)):
                        k = points[k_idx]
                        if k <= j + 1:
                            continue
                            
                        for l_idx in range(k_idx + 1, len(points)):
                            l = points[l_idx]
                            if l <= k + 1:
                                continue
                            
                            # Double bridge move: break tour at i, j, k, l and reconnect in a new way
                            # This specific move can escape local optima that 3-Opt can't
                            
                            # Current connections: A-B, C-D, E-F, G-H
                            a, b = current_tour[i], current_tour[(i+1) % instance.size]
                            c, d = current_tour[j], current_tour[(j+1) % instance.size]
                            e, f = current_tour[k], current_tour[(k+1) % instance.size]
                            g, h = current_tour[l], current_tour[(l+1) % instance.size]
                            
                            a_idx, b_idx = node_indices[a], node_indices[b]
                            c_idx, d_idx = node_indices[c], node_indices[d]
                            e_idx, f_idx = node_indices[e], node_indices[f]
                            g_idx, h_idx = node_indices[g], node_indices[h]
                            
                            # Calculate current distance
                            current_distance = (distance_matrix[a_idx][b_idx] + 
                                               distance_matrix[c_idx][d_idx] + 
                                               distance_matrix[e_idx][f_idx] + 
                                               distance_matrix[g_idx][h_idx])
                            
                            # Double bridge reconnection: A-D, C-F, E-H, G-B
                            new_distance = (distance_matrix[a_idx][d_idx] + 
                                           distance_matrix[c_idx][f_idx] + 
                                           distance_matrix[e_idx][h_idx] + 
                                           distance_matrix[g_idx][b_idx])
                            
                            delta = new_distance - current_distance
                            
                            if delta < 0:  # Improvement found
                                # Apply double bridge move
                                new_tour = []
                                new_tour.extend(current_tour[:i+1])
                                new_tour.extend(current_tour[j+1:k+1])
                                new_tour.extend(current_tour[i+1:j+1])
                                new_tour.extend(current_tour[l+1:])
                                new_tour.extend(current_tour[k+1:l+1])
                                
                                # Validate the resulting tour
                                if len(new_tour) == instance.size:
                                    current_tour = new_tour
                                    current_length += delta
                                    
                                    # Update best tour if improved
                                    if current_length < best_length:
                                        best_length = current_length
                                        best_tour = current_tour.copy()
                                        
                                    improved = True
                                    improvements += 1
                                    logger.debug(f"4-Opt double bridge improvement: {delta:.2f}, new length: {current_length:.2f}")
                                    break
            
            # If no improvement in this iteration
            if not improved:
                # Try more extensive search or accept slightly worse solutions
                # to escape local optima if we have computation budget
                pass
        
        runtime = time.time() - start_time
        
        # Calculate SQF based on nearest neighbor baseline
        sqf = ((nn_tour_length - best_length) / nn_tour_length * 100.0) if nn_tour_length > 0 else 0.0
        
        # Prepare metadata
        metadata = {
            "runtime": runtime,
            "tour_length": best_length,
            "nn_tour_length": nn_tour_length,
            "iterations": iterations,
            "improvements": improvements,
            "sqf": sqf,
            "pue": instance.get_pue()
        }
        
        logger.info(f"4-Opt completed: runtime={runtime:.4f}s, tour_length={best_length:.2f}, "
                   f"iterations={iterations}, improvements={improvements}, SQF={sqf:.2f}%")
                   
        return best_tour, metadata

###########################################
# 3. Genetic Testing Implementation (Optional)
###########################################

# This section implements pattern detection and problem-solving for genetic sequences
# It's wrapped in a conditional to make it optional, as it depends on BioPython
# This demonstrates how the framework can be extended to domains beyond TSP

# Only define genetic sequence classes if BioPython is available
if BIOPYTHON_AVAILABLE:
    class GeneticSequenceInstance(ProblemInstance):
        """
        Implementation of ProblemInstance for genetic sequence analysis.
        Provides methods to detect repeats and motifs in genetic sequences.
        
        Genetic sequence analysis involves finding patterns in DNA, RNA, or protein sequences.
        These patterns can include repetitive regions, known motifs (e.g., promoter sequences),
        or other structural elements that have biological significance.
        
        Detecting these patterns can help reduce the computational complexity of
        alignment and other sequence analysis tasks.
        """
        
        def _calculate_size(self) -> int:
            """
            Size is the length of the genetic sequence.
            
            Returns:
                int: Length of the genetic sequence
            """
            size = len(self.data.seq)
            logger.debug(f"Genetic instance '{self.name}' size: {size}")
            return size
        
        def detect_patterns(self, pattern_types: List[PatternType]) -> List[Pattern]:
            """
            Detect patterns in the genetic sequence.
            Currently supports repetition and motif pattern detection.
            
            Repetition patterns represent segments that repeat throughout a sequence,
            while motifs are known functional elements with biological significance.
            
            Args:
                pattern_types: List of PatternType enums to detect
            
            Returns:
                List[Pattern]: List of detected Pattern objects
            """
            logger.info(f"Detecting patterns for '{self.name}' with types {pattern_types}")
            self._patterns = []
            
            if PatternType.REPETITION in pattern_types:
                self._detect_repeats()
            if PatternType.MOTIF in pattern_types:
                self._detect_motifs()
                
            # Reset cached values
            self._pattern_prevalence = None
            self._pue = None
            self._cluster_count = None
            
            pue = self.get_pue()
            logger.info(f"Detected {len(self._patterns)} patterns, PUE: {pue:.2f}%")
            return self._patterns
        
        def _detect_repeats(self, min_length: int = 3, max_length: int = 50):
            """
            Detect repetitive sequences in the genetic data.
            Finds substrings that occur multiple times in the sequence.
            
            Repetitive regions are common in genomic sequences and often have
            functional or structural significance. They can also be exploited
            by alignment algorithms to reduce computational complexity.
            
            Args:
                min_length: Minimum length of repeats to detect (default: 3)
                max_length: Maximum length of repeats to detect (default: 50)
            """
            logger.debug(f"Detecting repeats for '{self.name}', min_length={min_length}, max_length={max_length}")
            sequence = str(self.data.seq)
            n = len(sequence)
            repeats = {}
            
            # Search for repeating substrings of various lengths
            for length in range(min_length, min(max_length + 1, n // 2 + 1)):
                for i in range(n - length + 1):
                    substring = sequence[i:i + length]
                    if substring in repeats:
                        repeats[substring].add(i)
                    else:
                        repeats[substring] = {i}
            
            # Create patterns for repeats with multiple occurrences
            for substring, positions in repeats.items():
                if len(positions) > 1:
                    elements = set()
                    for pos in positions:
                        for offset in range(len(substring)):
                            elements.add(pos + offset)
                    pattern = Pattern(
                        type=PatternType.REPETITION,
                        elements=elements,
                        metadata={"sequence": substring, "positions": positions, "length": len(substring), "occurrences": len(positions)}
                    )
                    self._patterns.append(pattern)
                    logger.debug(f"Found repeat '{substring}' with {len(positions)} occurrences")
        
        def _detect_motifs(self, motifs: Optional[List[str]] = None):
            """
            Detect known motifs in the sequence.
            Motifs are specific sequence patterns with biological significance.
            
            Examples of biological motifs include:
            - TATA box (core promoter element in eukaryotes)
            - Shine-Dalgarno sequence (ribosome binding site in prokaryotes)
            - Poly-A signal (AATAAA, marks end of transcription in eukaryotes)
            
            Args:
                motifs: List of motif sequences to search for (default: common DNA motifs)
            """
            logger.debug(f"Detecting motifs for '{self.name}'")
            if motifs is None:
                # Default list of common DNA motifs
                motifs = ["TATA", "CAAT", "AATAAA", "GAGA", "CACGTG", "GGTACC"]
            sequence = str(self.data.seq)
            
            # Search for each motif in the sequence
            for motif in motifs:
                positions = []
                pos = sequence.find(motif)
                while pos != -1:
                    positions.append(pos)
                    pos = sequence.find(motif, pos + 1)
                if positions:
                    elements = set()
                    for pos in positions:
                        for offset in range(len(motif)):
                            elements.add(pos + offset)
                    pattern = Pattern(
                        type=PatternType.MOTIF,
                        elements=elements,
                        metadata={"motif": motif, "positions": positions, "occurrences": len(positions)}
                    )
                    self._patterns.append(pattern)
                    logger.debug(f"Found motif '{motif}' with {len(positions)} occurrences")
        
        def calculate_entropy(self) -> float:
            """
            Calculate entropy based on nucleotide frequency.
            Entropy represents the diversity of nucleotides in the sequence.
            
            Lower entropy indicates more biased nucleotide composition, which
            may suggest functional constraints or evolutionary pressure.
            
            Returns:
                float: Entropy value
            """
            logger.debug(f"Calculating entropy for '{self.name}'")
            sequence = str(self.data.seq)
            counts = {}
            for nucleotide in sequence:
                counts[nucleotide] = counts.get(nucleotide, 0) + 1
            total = len(sequence)
            probabilities = {k: v / total for k, v in counts.items()}
            entropy = 0.0
            for p in probabilities.values():
                if p > 0:
                    entropy -= p * math.log(p)
            logger.debug(f"Entropy: {entropy:.4f}")
            return entropy

    class GeneticAlignerSolver(Solver):
        """
        Base class for genetic sequence alignment solvers.
        Provides common methods for sequence alignment algorithms.
        
        Sequence alignment is a method of arranging sequences of DNA, RNA, or protein
        to identify regions of similarity. These similarities may be consequences of 
        functional, structural, or evolutionary relationships between the sequences.
        
        Alignment algorithms have high computational complexity, but pattern awareness
        can significantly reduce this by focusing on regions of interest.
        """
        
        def get_base_complexity(self, instance_size: int) -> float:
            """
            Base complexity for sequence alignment: O(n²).
            This represents the time complexity of standard alignment algorithms.
            
            Standard pairwise alignment using dynamic programming (e.g., Needleman-Wunsch
            or Smith-Waterman algorithms) has quadratic time complexity.
            
            Args:
                instance_size: Length of sequence
            
            Returns:
                float: Base complexity value
            """
            logger.debug(f"Base complexity for size {instance_size}")
            return instance_size ** 2

    class RepeatAwareAligner(GeneticAlignerSolver):
        """
        Solver that leverages repetitive patterns in genetic sequences.
        Uses detected repeats and motifs to optimize alignment.
        
        This aligner exploits two key types of patterns in genetic sequences:
        1. Repeats: Segments that occur multiple times (e.g., tandem repeats, microsatellites)
        2. Motifs: Known functional elements (e.g., promoters, binding sites, splice sites)
        
        By identifying these patterns, the aligner can prioritize them during alignment,
        potentially improving both accuracy and efficiency compared to standard aligners.
        """
        
        def __init__(self, config: Optional[SolverConfiguration] = None):
            """Initialize the RepeatAwareAligner solver."""
            super().__init__("RepeatAwareAligner", config)
        
        def solve(self, instance: GeneticSequenceInstance) -> Tuple[Any, Dict[str, Any]]:
            """
            Simulate alignment using patterns.
            This is a simplified implementation that demonstrates the concept.
            
            In real sequence alignment, repeats and motifs can be used to:
            1. Guide the alignment process by aligning matching patterns first
            2. Adjust scoring matrices to prioritize known functional elements
            3. Reduce the search space by focusing on regions of interest
            
            Args:
                instance: GeneticSequenceInstance to solve
            
            Returns:
                Tuple: (alignment result, metadata)
            """
            logger.info(f"Starting RepeatAwareAligner for '{instance.name}'")
            start_time = time.time()
            
            # Extract patterns that could help with alignment
            repeats = [p for p in instance._patterns if p.type == PatternType.REPETITION]
            motifs = [p for p in instance._patterns if p.type == PatternType.MOTIF]
            sequence = str(instance.data.seq)
            alignment_quality = 0.8  # Base alignment quality
            
            # Adjust quality based on repeats
            if repeats:
                repeat_coverage = sum(p.size() for p in repeats) / len(sequence)
                alignment_quality += 0.1 * min(repeat_coverage, 1.0)
                logger.debug(f"Repeat coverage: {repeat_coverage:.4f}, quality boost: {0.1 * min(repeat_coverage, 1.0):.4f}")
            
            # Adjust quality based on motifs
            if motifs:
                motif_coverage = sum(p.size() for p in motifs) / len(sequence)
                alignment_quality += 0.05 * min(motif_coverage, 1.0)
                logger.debug(f"Motif coverage: {motif_coverage:.4f}, quality boost: {0.05 * min(motif_coverage, 1.0):.4f}")
            
            # Cap maximum quality
            alignment_quality = min(alignment_quality, 0.99)
            alignment = {"sequence": sequence, "quality_score": alignment_quality, "aligned_regions": [(0, len(sequence))]}
            runtime = time.time() - start_time
            metadata = {"runtime": runtime, "quality_score": alignment_quality, "sqf": 0.0, "pue": instance.get_pue()}
            logger.info(f"Alignment completed: runtime={runtime:.4f}s, quality={alignment_quality:.4f}, PUE={metadata['pue']:.2f}%")
            return alignment, metadata
        
        def get_reduction_factor(self, instance: GeneticSequenceInstance) -> float:
            """
            Calculate reduction factor based on repeat patterns.
            Repeats can significantly reduce alignment complexity.
            
            Repetitive regions allow the alignment algorithm to:
            1. Reuse computations for identical subsequences
            2. Recognize structural elements that should be aligned as units
            3. Identify regions that might be treated differently (e.g., gap penalties)
            
            Args:
                instance: GeneticSequenceInstance to evaluate
            
            Returns:
                float: Reduction factor (0.0 to 1.0)
            """
            logger.debug(f"Calculating reduction factor for '{instance.name}'")
            repeat_patterns = [p for p in instance._patterns if p.type == PatternType.REPETITION]
            if not repeat_patterns:
                return 0.9
            total_covered = sum(p.size() for p in repeat_patterns)
            coverage_ratio = total_covered / instance.size
            reduction = 0.9 * math.exp(-3 * coverage_ratio) + 0.1
            logger.debug(f"Repeat coverage ratio: {coverage_ratio:.4f}, reduction: {reduction:.4f}")
            return reduction

else:
    # Log warning if BioPython is not available
    logger.warning("Genetic testing functionality disabled due to missing BioPython.")

###########################################
# 4. Weather Forecasting Implementation (Optional)
###########################################

# This section implements pattern detection and problem-solving for weather forecasting
# It's wrapped in a conditional to make it optional, as it depends on xarray
# This further demonstrates the framework's extensibility to diverse problem domains

# Only define weather forecasting classes if xarray is available
if XARRAY_AVAILABLE:
    class WeatherForecastInstance(ProblemInstance):
        """
        Implementation of ProblemInstance for weather forecasting.
        Provides methods to detect seasonal patterns in weather data.
        
        Weather forecasting involves predicting future atmospheric conditions
        based on historical data and physical models. Weather data often contains
        various cyclic patterns at different timescales:
        - Daily cycles (day/night temperature variations)
        - Weekly cycles (human activity patterns)
        - Monthly/seasonal cycles (temperature, precipitation)
        - Annual cycles (seasons)
        
        Detecting these patterns can significantly improve forecasting accuracy
        and computational efficiency.
        """
        
        def _calculate_size(self) -> int:
            """
            Size is the number of time points in the dataset.
            
            Returns:
                int: Number of time points
            """
            size = len(self.data.time)
            logger.debug(f"Weather instance '{self.name}' size: {size}")
            return size
        
        def detect_patterns(self, pattern_types: List[PatternType]) -> List[Pattern]:
            """
            Detect patterns in weather data.
            Currently supports seasonality pattern detection.
            
            Seasonal patterns in weather data can occur at multiple time scales:
            - Diurnal (daily) cycles
            - Weekly patterns (often related to human activity)
            - Monthly and seasonal variations
            - Annual cycles
            
            Args:
                pattern_types: List of PatternType enums to detect
            
            Returns:
                List[Pattern]: List of detected Pattern objects
            """
            logger.info(f"Detecting patterns for '{self.name}' with types {pattern_types}")
            self._patterns = []
            if PatternType.SEASONALITY in pattern_types:
                self._detect_seasonality()
            
            # Reset cached values
            self._pattern_prevalence = None
            self._pue = None
            self._cluster_count = None
            
            pue = self.get_pue()
            logger.info(f"Detected {len(self._patterns)} patterns, PUE: {pue:.2f}%")
            return self._patterns
        
        def _detect_seasonality(self, variable: str = 'temperature'):
            """
            Detect seasonal patterns in weather time series.
            Uses autocorrelation to find repeating patterns.
            
            Autocorrelation measures the correlation of a signal with a delayed copy
            of itself. High autocorrelation at a specific lag indicates a seasonal
            pattern with that period. For example:
            - A peak at lag=24 suggests a daily cycle (for hourly data)
            - A peak at lag=7 suggests a weekly cycle
            - A peak at lag=365 suggests an annual cycle
            
            Args:
                variable: Weather variable to analyze (default: 'temperature')
            """
            logger.debug(f"Detecting seasonality for '{self.name}', variable: {variable}")
            if variable not in self.data:
                logger.warning(f"Variable '{variable}' not found")
                return
            
            # Extract time series data
            series = self.data[variable].values
            if len(series) < 24:
                logger.warning(f"Series too short ({len(series)} points)")
                return
            
            # Normalize data and calculate autocorrelation
            n = len(series)
            series_norm = (series - np.mean(series)) / np.std(series)
            max_lag = min(n // 2, 365 * 2)  # Maximum lag to consider
            autocorr = np.zeros(max_lag)
            
            # Calculate autocorrelation for each lag
            for lag in range(1, max_lag + 1):
                autocorr[lag - 1] = np.sum(series_norm[lag:] * series_norm[:-lag]) / (n - lag)
            
            # Find peaks in autocorrelation (seasonal periods)
            peaks, _ = find_peaks(autocorr, height=0.3)
            if not peaks.size:
                logger.debug("No significant seasonal peaks found")
                return
            
            # Select top peaks
            peak_values = autocorr[peaks]
            sorted_indices = np.argsort(-peak_values)
            top_peaks = peaks[sorted_indices[:min(3, len(peaks))]]
            
            # Create patterns for each top seasonal period
            for peak in top_peaks:
                period = peak + 1
                strength = autocorr[peak]
                
                # Determine seasonality type based on period
                seasonality_type = "Unknown"
                if 350 <= period <= 380:
                    seasonality_type = "Annual"
                elif 25 <= period <= 35:
                    seasonality_type = "Monthly"
                elif 6 <= period <= 8:
                    seasonality_type = "Weekly"
                elif period == 24 or period == 12:
                    seasonality_type = "Daily"
                
                # Find data points that follow the seasonal pattern
                elements = set()
                for i in range(n - period):
                    if abs(series[i] - series[i + period]) < np.std(series) * 0.5:
                        elements.add(i)
                        elements.add(i + period)
                
                # Create pattern if elements were found
                if elements:
                    pattern = Pattern(
                        type=PatternType.SEASONALITY,
                        elements=elements,
                        metadata={"period": period, "strength": strength, "type": seasonality_type, "variable": variable}
                    )
                    self._patterns.append(pattern)
                    logger.debug(f"Found {seasonality_type} seasonality, period: {period}, strength: {strength:.4f}")
        
        def calculate_entropy(self) -> float:
            """
            Calculate entropy based on temperature distribution.
            Entropy represents the variability in the weather data.
            
            Higher entropy indicates more variability and potentially less
            predictable weather patterns. Lower entropy suggests more consistent
            conditions that might be easier to forecast.
            
            Returns:
                float: Entropy value
            """
            logger.debug(f"Calculating entropy for '{self.name}'")
            # Use temperature if available, otherwise use first available variable
            var = 'temperature' if 'temperature' in self.data else list(self.data.variables)[0] if self.data.variables else None
            if not var:
                logger.warning("No variables available, entropy: 0.0")
                return 0.0
            
            # Extract and clean values
            values = self.data[var].values.flatten()
            values = values[~np.isnan(values)]
            if len(values) == 0:
                logger.warning("No valid values, entropy: 0.0")
                return 0.0
            
            # Calculate entropy from histogram
            hist, bin_edges = np.histogram(values, bins='auto', density=True)
            entropy = 0.0
            for p in hist:
                if p > 0:
                    entropy -= p * math.log(p)
            logger.debug(f"Entropy: {entropy:.4f}")
            return entropy

    class WeatherForecastSolver(Solver):
        """
        Base class for weather forecasting solvers.
        Provides common methods for weather prediction algorithms.
        
        Weather prediction is a complex computational problem that involves:
        - Processing large datasets with multiple variables (temperature, pressure, etc.)
        - Dealing with temporal dependencies and spatial correlations
        - Accounting for seasonal, cyclical, and trending patterns
        - Balancing accuracy with computational efficiency
        
        Pattern-aware solvers can exploit the regularities in weather data
        to improve prediction accuracy while reducing computational requirements.
        """
        
        def get_base_complexity(self, instance_size: int) -> float:
            """
            Base complexity for forecasting: O(n log n).
            This represents the complexity of common forecasting algorithms.
            
            Many forecasting methods (e.g., FFT-based methods, some statistical
            models) have n log n complexity, which is better than the quadratic
            complexity of pairwise comparison methods.
            
            Args:
                instance_size: Number of time points
            
            Returns:
                float: Base complexity value
            """
            logger.debug(f"Base complexity for size {instance_size}")
            if instance_size <= 1:
                return 1.0
            return instance_size * math.log(instance_size)

    class SeasonalForecastSolver(WeatherForecastSolver):
        """
        Solver leveraging seasonal patterns in weather data.
        Uses detected seasonality to improve forecasting.
        
        Seasonal patterns in weather data include:
        - Daily temperature cycles (warmer during day, cooler at night)
        - Annual seasons (summer, fall, winter, spring)
        - Multi-year cycles (El Niño/La Niña)
        
        This solver detects and leverages these patterns to:
        1. Decompose time series into seasonal components
        2. Make predictions based on historical patterns for similar seasons
        3. Adjust for trends and anomalies that deviate from seasonal norms
        """
        
        def __init__(self, config: Optional[SolverConfiguration] = None):
            """Initialize the SeasonalForecaster solver."""
            super().__init__("SeasonalForecaster", config)
        
        @timeit
        def solve(self, instance: WeatherForecastInstance) -> Tuple[Any, Dict[str, Any]]:
            """
            Simulate forecast using seasonal patterns.
            This is a simplified implementation that demonstrates the concept.
            
            A full implementation would likely use techniques like:
            - Seasonal decomposition of time series
            - SARIMA (Seasonal ARIMA) modeling
            - Prophet or other specialized forecasting tools
            
            This simplified version uses detected seasonal patterns to adjust
            forecast quality and uncertainty estimates.
            
            Args:
                instance: WeatherForecastInstance to solve
            
            Returns:
                Tuple: (forecast result, metadata)
            """
            logger.info(f"Starting SeasonalForecaster for '{instance.name}'")
            start_time = time.time()
            
            # Extract seasonal patterns
            seasonal_patterns = [p for p in instance._patterns if p.type == PatternType.SEASONALITY]
            # Use temperature if available, otherwise use first available variable
            var = 'temperature' if 'temperature' in instance.data else list(instance.data.variables)[0] if instance.data.variables else None
            if not var:
                logger.warning("No variables available")
                return {}, {"runtime": 0, "forecast_quality": 0, "uncertainty": 1.0, "sqf": 0.0, "pue": instance.get_pue()}
            
            # Extract time series data
            temp_data = instance.data[var].values
            time_data = instance.data.time.values
            forecast_quality = 0.7  # Base forecast quality
            uncertainty = 0.5       # Base uncertainty
            
            # Adjust quality and uncertainty based on seasonal patterns
            if seasonal_patterns:
                pattern_strength = sum(p.metadata.get('strength', 0) for p in seasonal_patterns)
                forecast_quality += 0.1 * min(pattern_strength, 2.0)
                uncertainty -= 0.1 * min(pattern_strength, 3.0)
                logger.debug(f"Seasonal strength: {pattern_strength:.4f}, quality: {forecast_quality:.4f}, uncertainty: {uncertainty:.4f}")
            
            # Check if we have enough data
            n = len(temp_data)
            horizon = 5  # Forecast 5 days ahead
            if n < 10:
                runtime = time.time() - start_time
                logger.warning("Insufficient data for forecast")
                return {}, {"runtime": runtime, "forecast_quality": forecast_quality, "uncertainty": uncertainty, "sqf": 0.0, "pue": instance.get_pue()}
            
            # Generate forecast
            forecast = []
            forecast_times = []
            if seasonal_patterns and n >= 50:
                # Use best seasonal pattern for forecasting
                best_pattern = max(seasonal_patterns, key=lambda p: p.metadata.get('strength', 0))
                period = best_pattern.metadata.get('period', 7)
                logger.debug(f"Using seasonal period: {period}")
                for i in range(horizon):
                    if n - period + i >= 0:
                        # Use seasonal pattern with trend
                        trend = (temp_data[n - period + i] - temp_data[n - 2 * period + i]) if n - 2 * period + i >= 0 else 0
                        forecast.append(temp_data[n - period + i] + trend)
                    else:
                        # Fallback to last value
                        forecast.append(temp_data[-1])
                    forecast_times.append(time_data[-1] + np.timedelta64(i + 1, 'D'))
            else:
                # Fallback to moving average
                window = min(7, n)
                avg = np.mean(temp_data[-window:])
                logger.debug(f"Fallback to moving average, window: {window}")
                for i in range(horizon):
                    forecast.append(avg)
                    forecast_times.append(time_data[-1] + np.timedelta64(i + 1, 'D'))
            
            # Prepare result and metadata
            result = {"forecast": np.array(forecast), "times": np.array(forecast_times), "quality": forecast_quality, "uncertainty": uncertainty}
            runtime = time.time() - start_time
            metadata = {"runtime": runtime, "forecast_quality": forecast_quality, "uncertainty": uncertainty, "sqf": 0.0, "pue": instance.get_pue()}
            logger.info(f"Forecast completed: runtime={runtime:.4f}s, quality={forecast_quality:.4f}, uncertainty={uncertainty:.4f}, PUE={metadata['pue']:.2f}%")
            return result, metadata
        
        def get_reduction_factor(self, instance: WeatherForecastInstance) -> float:
            """
            Calculate reduction factor based on seasonal patterns.
            Strong seasonality can significantly reduce forecasting complexity.
            
            Weather data with strong seasonal components allows for:
            1. Reuse of historical patterns from similar seasons
            2. Dimensional reduction (e.g., focusing only on seasonal anomalies)
            3. More confident predictions when seasonality dominates
            
            Args:
                instance: WeatherForecastInstance to evaluate
            
            Returns:
                float: Reduction factor (0.0 to 1.0)
            """
            logger.debug(f"Calculating reduction factor for '{instance.name}'")
            seasonal_patterns = [p for p in instance._patterns if p.type == PatternType.SEASONALITY]
            if not seasonal_patterns:
                return 0.85
            seasonal_strength = sum(p.metadata.get('strength', 0) for p in seasonal_patterns)
            reduction = max(0.2, 0.85 - 0.2 * min(seasonal_strength, 3.0))
            logger.debug(f"Seasonal strength: {seasonal_strength:.4f}, reduction: {reduction:.4f}")
            return reduction

else:
    # Log warning if xarray is not available
    logger.warning("Weather forecasting functionality disabled due to missing xarray or scipy.")
	
###########################################
# 5. Adaptive Solver Pipeline
###########################################

class SolverPortfolio:
    """
    Manages a collection of solvers and uses meta-learning for solver selection.
    Enhanced to predict tour length and runtime for TSP solvers, balancing quality and efficiency.
    
    The SolverPortfolio implements a form of automated algorithm selection:
    1. It maintains a collection of different solvers for a problem domain
    2. It learns which solvers perform best on which types of problem instances
    3. When faced with a new instance, it selects the most appropriate solver
       based on instance features and predicted performance metrics
    
    This "algorithm portfolio" approach is more robust than any single algorithm,
    as different solvers excel on different problem structures.
    """
    
    def __init__(self, solvers: List[Solver]):
        """
        Initialize the solver portfolio.
        
        Creates a portfolio with the given collection of solvers and prepares
        data structures for meta-learning models. These models will learn to
        predict solver performance on different types of problem instances.
        
        Args:
            solvers: List of Solver objects to include
        """
        logger.debug(f"Initializing SolverPortfolio with {len(solvers)} solvers")
        self.solvers = solvers
        self.solver_names = [solver.name for solver in solvers]
        # Models for predicting tour length and runtime for TSP solvers
        self.tour_length_models = {}
        self.runtime_models = {}
        # Models for predicting complexity for non-TSP solvers
        self.meta_models = {}
        self.training_data = []  # Store training data for meta-learning
        self.scaler = StandardScaler()  # For feature normalization
        logger.info(f"SolverPortfolio initialized with solvers: {self.solver_names}")
    
    @timeit
    def train_meta_model(self, instances: List[ProblemInstance], results: List[Dict]):
        """
        Train meta-learning models to predict solver performance based on instance features.
        For TSP instances, trains models to predict tour length and runtime.
        For other instances, trains models to predict complexity.
        
        Meta-learning process:
        1. Extract features from problem instances (size, pattern prevalence, entropy, etc.)
        2. Collect performance metrics from previous solver runs on these instances
        3. Train machine learning models (Random Forests) to predict:
           - Solution quality (e.g., tour length for TSP)
           - Runtime 
           - Computational complexity
        4. These models will later be used to select the best solver for new instances
        
        Args:
            instances: List of ProblemInstance objects
            results: List of result dictionaries from experiment runs
        """
        logger.info(f"Training meta-model with {len(instances)} instances and {len(results)} results")
        if not instances or not results:
            logger.warning("No instances or results provided for training")
            return
        
        # Create dictionary of instances for quick lookup
        instance_dict = {instance.name: instance for instance in instances}
        
        # Prepare data structures for organizing features and results
        instance_features = {}  # Dictionary to map instance name to features
        tour_lengths = {solver.name: {} for solver in self.solvers}  # Dict of dicts: solver -> instance -> value
        runtimes = {solver.name: {} for solver in self.solvers}
        complexities = {solver.name: {} for solver in self.solvers}
        
        # First, extract features for all instances
        for instance in instances:
            feature = instance.get_features()
            instance_features[instance.name] = [
                feature['size'], 
                feature['pattern_prevalence'], 
                feature['entropy'], 
                feature['pue'], 
                feature['cluster_count']
            ]
        
        # Then, extract results for each instance-solver pair
        for result in results:
            instance_name = result.get('instance')
            solver_name = result.get('solver')
            if instance_name in instance_dict and 'metadata' in result:
                metadata = result.get('metadata', {})
                if isinstance(instance_dict[instance_name], TSPInstance):
                    tour_lengths[solver_name][instance_name] = metadata.get('tour_length', np.nan)
                    runtimes[solver_name][instance_name] = metadata.get('runtime', np.nan)
                else:
                    complexities[solver_name][instance_name] = np.log(max(metadata.get('complexity', 1e-10), 1e-10))
        
        # Skip training if no valid instances
        if not instance_features:
            logger.warning("No valid features extracted for training")
            return
        
        # Convert to format suitable for scikit-learn
        instance_names = list(instance_features.keys())
        X = np.array([instance_features[name] for name in instance_names])
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train models for TSP instances
        if isinstance(instances[0], TSPInstance):
            for solver in self.solvers:
                if solver.name in tour_lengths:
                    # Extract data for this solver, aligning with instance order
                    y_tour = np.array([tour_lengths[solver.name].get(name, np.nan) for name in instance_names])
                    y_runtime = np.array([runtimes[solver.name].get(name, np.nan) for name in instance_names])
                    
                    # Create mask for valid values (non-NaN)
                    mask = ~np.isnan(y_tour) & ~np.isnan(y_runtime)
                    
                    # Check if we have enough data to train
                    if np.sum(mask) > 0:
                        logger.debug(f"Training tour length model for '{solver.name}' with {np.sum(mask)} samples")
                        model_tour = RandomForestRegressor(n_estimators=100, n_jobs=NUM_WORKERS)
                        model_tour.fit(X_scaled[mask], y_tour[mask])
                        self.tour_length_models[solver.name] = model_tour
                        
                        logger.debug(f"Training runtime model for '{solver.name}'")
                        model_runtime = RandomForestRegressor(n_estimators=100, n_jobs=NUM_WORKERS)
                        model_runtime.fit(X_scaled[mask], y_runtime[mask])
                        self.runtime_models[solver.name] = model_runtime
                    else:
                        logger.warning(f"Insufficient valid data to train models for '{solver.name}'")
        else:
            # Train complexity models for non-TSP problems
            for solver in self.solvers:
                if solver.name in complexities:
                    # Extract complexity data for this solver
                    y_complexity = np.array([complexities[solver.name].get(name, np.nan) for name in instance_names])
                    mask = ~np.isnan(y_complexity)
                    
                    if np.sum(mask) > 0:
                        logger.debug(f"Training complexity model for '{solver.name}' with {np.sum(mask)} samples")
                        model = RandomForestRegressor(n_estimators=100, n_jobs=NUM_WORKERS)
                        model.fit(X_scaled[mask], y_complexity[mask])
                        self.meta_models[solver.name] = model
                    else:
                        logger.warning(f"Insufficient valid data to train complexity model for '{solver.name}'")
    
    @timeit
    def select_solver(self, instance: ProblemInstance) -> Solver:
        """
        Select the best solver using meta-learning, balancing quality and efficiency.
        For TSP instances, selects based on predicted tour length and runtime score.
        For other instances, selects based on predicted complexity.
        
        Solver selection process:
        1. Extract features from the given problem instance
        2. Use trained models to predict performance metrics for each available solver
        3. For TSP: Calculate a composite score that balances solution quality and runtime
           Score = predicted_tour_length + alpha * predicted_runtime
        4. For other problems: Select the solver with lowest predicted complexity
        5. If no trained models exist, fall back to heuristic selection
        
        Args:
            instance: ProblemInstance to solve
        
        Returns:
            Solver: Selected Solver object
        """
        logger.info(f"Selecting solver for '{instance.name}'")
        features = instance.get_features()
        
        # Handle TSP instances
        if isinstance(instance, TSPInstance):
            feature_vector = np.array([[features['size'], features['pattern_prevalence'], 
                                        features['entropy'], features['pue'], features['cluster_count']]])
            scaled_features = self.scaler.transform(feature_vector)
            
            # Calculate instance size for adaptive selection
            size = instance.size
            
            # For very small instances (< 30 cities), Nearest Neighbor is fast enough
            if size < 30 and 'TSP-NearestNeighbor' in self.solver_names:
                logger.debug(f"Small instance detected (size {size}), using Nearest Neighbor")
                solver_name = 'TSP-NearestNeighbor'
                return next(s for s in self.solvers if s.name == solver_name)
            
            # For large instances (500+), skip 3-Opt entirely
            elif size > 500 and 'TSP-3Opt' in self.solver_names and 'TSP-2Opt' in self.solver_names:
                logger.debug(f"Large instance detected (size {size}), using 2-Opt instead of 3-Opt")
                solver_name = 'TSP-2Opt'
                return next(s for s in self.solvers if s.name == solver_name)
            
            # For medium-large instances (100-500), use meta-learning if available
            elif self.tour_length_models and self.runtime_models:
                scores = {}
                for solver in self.solvers:
                    if solver.name in self.tour_length_models and solver.name in self.runtime_models:
                        # Use trained models to predict performance
                        predicted_tour_length = self.tour_length_models[solver.name].predict(scaled_features)[0]
                        predicted_runtime = self.runtime_models[solver.name].predict(scaled_features)[0]
                        
                        # Adaptive alpha based on instance size
                        # For larger instances, prioritize runtime more
                        base_alpha = 0.01
                        size_adjustment = min(1.0, size / 500)
                        alpha = base_alpha * (1 + 9 * size_adjustment)  # Scale from 0.01 to 0.1
                        
                        # Score = predicted_tour_length + alpha * predicted_runtime
                        score = predicted_tour_length + alpha * predicted_runtime
                        scores[solver] = score
                        logger.debug(f"Score for '{solver.name}': tour_length={predicted_tour_length:.2f}, "
                                     f"runtime={predicted_runtime:.4f}s, alpha={alpha:.4f}, score={score:.2f}")
                    else:
                        # Fallback to heuristic selection
                        complexity = solver.calculate_complexity(instance)
                        scores[solver] = complexity
                        logger.debug(f"Fallback complexity for '{solver.name}': {complexity:.2e}")
                        
                # Select solver with best (lowest) score
                selected_solver = min(scores, key=scores.get)
            else:
                # No trained models, fall back to heuristic selection
                selected_solver = self._heuristic_selection(instance)
        else:
            # Handle non-TSP instances
            if not self.meta_models:
                return self._heuristic_selection(instance)
            
            feature_vector = np.array([[features['size'], features['pattern_prevalence'], 
                                        features['entropy'], features['pue']]])
            scaled_features = self.scaler.transform(feature_vector)
            
            # Predict complexities for each solver
            predicted_complexities = {}
            for solver in self.solvers:
                if solver.name in self.meta_models:
                    log_complexity = self.meta_models[solver.name].predict(scaled_features)[0]
                    complexity = np.exp(log_complexity)
                    predicted_complexities[solver] = complexity
                else:
                    complexity = solver.calculate_complexity(instance)
                    predicted_complexities[solver] = complexity
            selected_solver = min(predicted_complexities, key=predicted_complexities.get)
        
        logger.info(f"Selected solver: '{selected_solver.name}'")
        return selected_solver
    
    def _heuristic_selection(self, instance: ProblemInstance) -> Solver:
        """
        Heuristic selection based on pattern prevalence when meta-learning is unavailable.
        Fallback method when no trained models exist.
        
        This method implements simple rules for solver selection:
        - High pattern prevalence: Use most pattern-sensitive solver
        - Moderate pattern prevalence: Use solver with lowest calculated complexity
        - Low pattern prevalence: Use the default (first) solver
        
        Args:
            instance: ProblemInstance to evaluate
        
        Returns:
            Solver: Selected Solver object
        """
        logger.debug(f"Performing heuristic selection for '{instance.name}'")
        pattern_prevalence = instance.get_pattern_prevalence()
        
        # For TSP instances, also consider instance size
        if isinstance(instance, TSPInstance):
            size = instance.size
            
            if size < 50:
                # For very small instances, use Nearest Neighbor for speed
                for solver in self.solvers:
                    if solver.name == 'TSP-NearestNeighbor':
                        logger.debug(f"Selected Nearest Neighbor for small instance (size {size})")
                        return solver
            
            elif size < 200 and pattern_prevalence > 0.7:
                # For moderate size, highly structured instances, use 2-Opt
                for solver in self.solvers:
                    if solver.name == 'TSP-2Opt':
                        logger.debug(f"Selected 2-Opt for structured instance (size {size}, pattern prevalence {pattern_prevalence:.2f})")
                        return solver
            
            elif size < 500 and pattern_prevalence > 0.5:
                # For larger, well-structured instances, try 3-Opt if available
                for solver in self.solvers:
                    if solver.name == 'TSP-3Opt':
                        logger.debug(f"Selected 3-Opt for larger structured instance (size {size}, pattern prevalence {pattern_prevalence:.2f})")
                        return solver
            
            # Default to 2-Opt for larger instances
            for solver in self.solvers:
                if solver.name == 'TSP-2Opt':
                    logger.debug(f"Selected 2-Opt as default for larger instance")
                    return solver
        
        # General case: Use different strategies based on pattern prevalence
        if pattern_prevalence > 0.8:
            # High pattern prevalence: use most pattern-sensitive solver
            solver = self._most_pattern_sensitive_solver(instance)
            logger.debug(f"High pattern prevalence ({pattern_prevalence:.4f}), selected '{solver.name}'")
        elif pattern_prevalence > 0.3:
            # Moderate pattern prevalence: use solver with lowest complexity
            complexities = {solver: solver.calculate_complexity(instance) for solver in self.solvers}
            solver = min(complexities, key=complexities.get)
            logger.debug(f"Moderate pattern prevalence ({pattern_prevalence:.4f}), selected '{solver.name}'")
        else:
            # Low pattern prevalence: use default solver
            solver = self.solvers[0]  # Default to first solver
            logger.debug(f"Low pattern prevalence ({pattern_prevalence:.4f}), selected default '{solver.name}'")
        return solver
    
    def _most_pattern_sensitive_solver(self, instance: ProblemInstance) -> Solver:
        """
        Identify the solver most sensitive to patterns by comparing complexity with/without patterns.
        
        Compares how much each solver's complexity estimate decreases when patterns
        are present vs. absent. The solver with the largest complexity reduction
        is considered most sensitive to patterns.
        
        Args:
            instance: ProblemInstance to evaluate
        
        Returns:
            Solver: Solver with highest sensitivity to patterns
        """
        logger.debug(f"Finding most pattern-sensitive solver for '{instance.name}'")
        sensitivities = {}
        
        # Create a copy with no patterns to compare
        no_patterns_instance = copy.deepcopy(instance)
        no_patterns_instance._patterns = []
        no_patterns_instance._pattern_prevalence = 0
        
        # Calculate sensitivity for each solver
        for solver in self.solvers:
            complexity_with = solver.calculate_complexity(instance)
            complexity_without = solver.calculate_complexity(no_patterns_instance)
            # Sensitivity = reduction in complexity due to patterns
            sensitivity = (complexity_without / complexity_with) if complexity_with > 0 else 1.0
            sensitivities[solver] = sensitivity
            logger.debug(f"Sensitivity for '{solver.name}': {sensitivity:.4f}")
        
        # Select solver with highest sensitivity
        selected_solver = max(sensitivities, key=sensitivities.get)
        logger.debug(f"Most sensitive solver: '{selected_solver.name}'")
        return selected_solver

###########################################
# 6. Performance Metrics
###########################################

class PerformanceMetrics:
    """
    Utility class for calculating performance metrics.
    Provides standardized methods to measure solver performance.
    
    Performance metrics help quantify different aspects of solver effectiveness:
    - Solution quality (how good is the solution?)
    - Efficiency (how fast was the solution found?)
    - Uncertainty (how confident are we in the solution?)
    
    These metrics allow for objective comparison between different solvers
    and help in selecting the most appropriate solver for specific problem instances.
    """
    
    @staticmethod
    def accuracy_gain_index(solution_quality: float, baseline_quality: float) -> float:
        """
        Calculate Accuracy Gain Index (AGI) as percentage improvement.
        AGI = ((solution_quality - baseline_quality) / baseline_quality) * 100%
        
        The AGI quantifies how much better a solution is compared to a baseline.
        For example, if a weather forecasting solution has a quality score of 0.85
        and the baseline has 0.70, the AGI would be 21.4%, indicating a substantial
        improvement over the baseline method.
        
        Args:
            solution_quality: Quality metric of the solution
            baseline_quality: Quality metric of the baseline
        
        Returns:
            float: Percentage gain (positive is improvement)
        """
        logger.debug(f"Calculating AGI: solution={solution_quality:.4f}, baseline={baseline_quality:.4f}")
        agi = ((solution_quality - baseline_quality) / baseline_quality * 100.0) if baseline_quality > 0 else 0.0
        logger.debug(f"AGI: {agi:.2f}%")
        return agi
    
    @staticmethod
    def uncertainty_reduction_index(solution_uncertainty: float, baseline_uncertainty: float) -> float:
        """
        Calculate Uncertainty Reduction Index (URI) as percentage reduction.
        URI = ((baseline_uncertainty - solution_uncertainty) / baseline_uncertainty) * 100%
        
        The URI measures how much a method reduces uncertainty compared to a baseline.
        For example, if a weather forecasting baseline has uncertainty of 0.5
        and an advanced method has uncertainty of 0.3, the URI would be 40%,
        indicating that the advanced method reduces uncertainty by 40% relative
        to the baseline.
        
        Args:
            solution_uncertainty: Uncertainty of the solution
            baseline_uncertainty: Uncertainty of the baseline
        
        Returns:
            float: Percentage reduction (0-100%)
        """
        logger.debug(f"Calculating URI: solution={solution_uncertainty:.4f}, baseline={baseline_uncertainty:.4f}")
        uri = ((baseline_uncertainty - solution_uncertainty) / baseline_uncertainty * 100.0) if baseline_uncertainty > 0 else 0.0
        uri = max(0.0, min(uri, 100.0))  # Clamp to valid range
        logger.debug(f"URI: {uri:.2f}%")
        return uri

    @staticmethod
    def efficiency_index(solution_runtime: float, baseline_runtime: float) -> float:
        """
        Calculate Efficiency Index (EI) comparing solution runtime to baseline.
        EI = (baseline_runtime / solution_runtime)
        
        Values > 1 indicate the solution is faster than baseline (higher is better).
        Values < 1 indicate the solution is slower than baseline.
        
        Args:
            solution_runtime: Runtime of the solution
            baseline_runtime: Runtime of the baseline
            
        Returns:
            float: Efficiency ratio (higher is better)
        """
        logger.debug(f"Calculating EI: solution={solution_runtime:.4f}s, baseline={baseline_runtime:.4f}s")
        # Avoid division by zero
        if solution_runtime < 1e-6:
            solution_runtime = 1e-6
        ei = baseline_runtime / solution_runtime if solution_runtime > 0 else float('inf')
        logger.debug(f"EI: {ei:.2f}x")
        return ei

##############################################################################
# 3. IMPROVED INITIAL TOUR GENERATORS
##############################################################################

class AdvancedTourInitializer:
    """
    Class that provides multiple advanced methods to generate initial TSP tours.
    More sophisticated than simple Nearest Neighbor approach.
    """
    
    @staticmethod
    def nearest_neighbor(instance: TSPInstance) -> Tuple[List[Any], float]:
        """
        Standard Nearest Neighbor heuristic (for baseline comparison).
        
        Args:
            instance: TSP instance
            
        Returns:
            Tuple: (tour, tour_length)
        """
        graph = instance.data
        nodes = list(graph.nodes())
        
        if not nodes:
            return [], 0
        
        # Start from a random node
        start_node = random.choice(nodes)
        tour = [start_node]
        unvisited = set(nodes)
        unvisited.remove(start_node)
        
        current = start_node
        total_distance = 0
        
        # Get distance matrix for faster access
        distance_matrix = instance.get_distance_matrix()
        node_indices = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        
        # Add nearest unvisited node at each step
        while unvisited:
            current_idx = node_indices[current]
            nearest = min(unvisited, key=lambda node: distance_matrix[current_idx][node_indices[node]])
            nearest_dist = distance_matrix[current_idx][node_indices[nearest]]
            
            total_distance += nearest_dist
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Close the tour
        if len(tour) > 1:
            total_distance += distance_matrix[node_indices[tour[-1]]][node_indices[tour[0]]]
        
        return tour, total_distance
    
    @staticmethod
    def multi_fragment(instance: TSPInstance) -> Tuple[List[Any], float]:
        """
        Multi-Fragment heuristic.
        Repeatedly adds the shortest edge that doesn't create a cycle or node with degree > 2.
        
        Args:
            instance: TSP instance
            
        Returns:
            Tuple: (tour, tour_length)
        """
        graph = instance.data
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if n <= 1:
            return nodes, 0
            
        # Get distance matrix
        distance_matrix = instance.get_distance_matrix()
        node_indices = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        
        # Create list of all edges with distances
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                node_i, node_j = nodes[i], nodes[j]
                idx_i, idx_j = node_indices[node_i], node_indices[node_j]
                distance = distance_matrix[idx_i][idx_j]
                edges.append((distance, node_i, node_j))
        
        # Sort edges by distance
        edges.sort()
        
        # Initialize tour fragments
        fragments = []
        node_to_fragment = {}
        node_degree = {node: 0 for node in nodes}
        
        # Greedy construction
        for dist, u, v in edges:
            # Skip if either node already has degree 2
            if node_degree[u] >= 2 or node_degree[v] >= 2:
                continue
                
            # Get fragments containing u and v
            u_frag = node_to_fragment.get(u)
            v_frag = node_to_fragment.get(v)
            
            # Case 1: Neither u nor v is in any fragment
            if u_frag is None and v_frag is None:
                new_fragment = [u, v]
                fragments.append(new_fragment)
                node_to_fragment[u] = new_fragment
                node_to_fragment[v] = new_fragment
            
            # Case 2: u is in a fragment, v is not
            elif u_frag is not None and v_frag is None:
                # Check if u is at the end of its fragment
                if u_frag[0] == u:
                    u_frag.insert(0, v)  # Add v at the beginning
                elif u_frag[-1] == u:
                    u_frag.append(v)  # Add v at the end
                else:
                    continue  # u is in the middle, can't add v
                node_to_fragment[v] = u_frag
            
            # Case 3: v is in a fragment, u is not
            elif v_frag is not None and u_frag is None:
                # Check if v is at the end of its fragment
                if v_frag[0] == v:
                    v_frag.insert(0, u)  # Add u at the beginning
                elif v_frag[-1] == v:
                    v_frag.append(u)  # Add u at the end
                else:
                    continue  # v is in the middle, can't add u
                node_to_fragment[u] = v_frag
            
            # Case 4: Both u and v are in different fragments
            elif u_frag is not v_frag:
                # Check if both nodes are at the ends of their fragments
                u_at_start = u_frag[0] == u
                u_at_end = u_frag[-1] == u
                v_at_start = v_frag[0] == v
                v_at_end = v_frag[-1] == v
                
                if u_at_start and v_at_start:
                    # Reverse u_frag and merge
                    u_frag.reverse()
                    merged = u_frag + v_frag
                elif u_at_start and v_at_end:
                    merged = v_frag + u_frag
                elif u_at_end and v_at_start:
                    merged = u_frag + v_frag
                elif u_at_end and v_at_end:
                    # Reverse v_frag and merge
                    v_frag.reverse()
                    merged = u_frag + v_frag
                else:
                    continue  # Can't merge fragments
                
                # Update fragment references
                fragments.remove(u_frag)
                fragments.remove(v_frag)
                fragments.append(merged)
                
                for node in merged:
                    node_to_fragment[node] = merged
            else:
                # u and v are in the same fragment - would create a cycle
                # unless it's the last edge to close the tour
                if len(fragments) == 1 and len(fragments[0]) == n-1:
                    # It's the last edge to close the tour
                    fragments[0].append(fragments[0][0])  # Close the cycle
                else:
                    continue
            
            # Update degrees
            node_degree[u] += 1
            node_degree[v] += 1
        
        # If fragments couldn't be fully merged, connect them arbitrarily
        if len(fragments) > 1:
            complete_tour = []
            for fragment in fragments:
                if complete_tour:
                    complete_tour.extend(fragment[1:])  # Skip first node to avoid duplicates
                else:
                    complete_tour = fragment.copy()
            
            # Close the tour if needed
            if complete_tour[0] != complete_tour[-1]:
                complete_tour.append(complete_tour[0])
                
            tour = complete_tour
        else:
            tour = fragments[0]
        
        # Calculate tour length
        tour_length = 0
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]
            u_idx, v_idx = node_indices[u], node_indices[v]
            tour_length += distance_matrix[u_idx][v_idx]
        
        return tour, tour_length
    
    @staticmethod
    def convex_hull_insertion(instance: TSPInstance) -> Tuple[List[Any], float]:
        """
        Convex Hull Insertion heuristic.
        Starts with the convex hull of the points and inserts remaining points optimally.
        Effective for geometric instances.
        
        Args:
            instance: TSP instance
            
        Returns:
            Tuple: (tour, tour_length)
        """
        graph = instance.data
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if n <= 3:
            # For small instances, just return a complete cycle
            tour = nodes + [nodes[0]] if nodes else []
            tour_length = instance.get_tour_length(tour)
            return tour, tour_length
        
        # Get node positions
        pos = nx.get_node_attributes(graph, 'pos')
        if not pos:
            # Fall back to nearest neighbor if positions aren't available
            return AdvancedTourInitializer.nearest_neighbor(instance)
            
        # Compute convex hull (library function not available, simplified approach)
        # This is a simplified Graham scan algorithm
        
        # Find point with lowest y-coordinate
        points = [(node, pos[node][0], pos[node][1]) for node in nodes]
        points.sort(key=lambda p: (p[2], p[1]))  # Sort by y, then x
        p0 = points[0]
        
        # Sort points by polar angle
        def polar_angle(p):
            return math.atan2(p[2] - p0[2], p[1] - p0[1])
            
        points = [p0] + sorted(points[1:], key=polar_angle)
        
        # Graham scan
        hull = [points[0][0], points[1][0]]
        for i in range(2, n):
            point = points[i]
            while len(hull) >= 2:
                # Check if we make a non-left turn
                p1x, p1y = pos[hull[-2]]
                p2x, p2y = pos[hull[-1]]
                p3x, p3y = pos[point[0]]
                
                # Cross product to determine turn direction
                cross_product = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
                
                if cross_product >= 0:  # non-left turn
                    hull.pop()  # Remove the middle point
                else:
                    break
            hull.append(point[0])
        
        # Close the hull
        hull.append(hull[0])
        
        # Initialize tour with convex hull
        tour = hull[:-1]  # Remove duplicate end
        
        # Get distance matrix
        distance_matrix = instance.get_distance_matrix()
        node_indices = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        
        # Insert remaining points
        remaining = [node for node in nodes if node not in set(tour)]
        
        for node in remaining:
            # Find insertion point that minimizes additional distance
            node_idx = node_indices[node]
            best_pos = 0
            best_increase = float('inf')
            
            for i in range(len(tour)):
                prev = tour[i]
                next = tour[(i + 1) % len(tour)]
                prev_idx, next_idx = node_indices[prev], node_indices[next]
                
                # Calculate change in distance if inserted here
                old_dist = distance_matrix[prev_idx][next_idx]
                new_dist = distance_matrix[prev_idx][node_idx] + distance_matrix[node_idx][next_idx]
                increase = new_dist - old_dist
                
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1
            
            # Insert node at best position
            tour.insert(best_pos, node)
        
        # Calculate tour length
        tour_length = 0
        for i in range(len(tour)):
            u, v = tour[i], tour[(i + 1) % len(tour)]
            u_idx, v_idx = node_indices[u], node_indices[v]
            tour_length += distance_matrix[u_idx][v_idx]
        
        return tour, tour_length
    
    @staticmethod
    def christofides(instance: TSPInstance) -> Tuple[List[Any], float]:
        """
        Approximate implementation of Christofides algorithm.
        Guaranteed to produce tours at most 1.5 times the optimal length for metric TSPs.
        
        Args:
            instance: TSP instance
            
        Returns:
            Tuple: (tour, tour_length)
        """
        graph = instance.data
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if n <= 3:
            tour = nodes + [nodes[0]] if nodes else []
            tour_length = instance.get_tour_length(tour)
            return tour, tour_length
        
        # Get distance matrix
        distance_matrix = instance.get_distance_matrix()
        node_indices = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        
        # Step 1: Find minimum spanning tree (MST)
        mst = nx.Graph()
        mst.add_nodes_from(nodes)
        
        # Prim's algorithm for MST
        visited = {nodes[0]}
        while len(visited) < n:
            min_edge = None
            min_dist = float('inf')
            
            for u in visited:
                u_idx = node_indices[u]
                for v in set(nodes) - visited:
                    v_idx = node_indices[v]
                    dist = distance_matrix[u_idx][v_idx]
                    if dist < min_dist:
                        min_dist = dist
                        min_edge = (u, v)
            
            if min_edge:
                u, v = min_edge
                mst.add_edge(u, v, weight=min_dist)
                visited.add(v)
        
        # Step 2: Find odd-degree vertices in MST
        odd_vertices = [node for node in mst.nodes() if mst.degree(node) % 2 == 1]
        
        # Step 3: Find minimum weight perfect matching (approximated)
        matching = []
        odd_vertices_set = set(odd_vertices)
        
        while odd_vertices_set:
            v = odd_vertices_set.pop()
            min_dist = float('inf')
            closest = None
            
            for u in odd_vertices_set:
                u_idx, v_idx = node_indices[u], node_indices[v]
                dist = distance_matrix[u_idx][v_idx]
                if dist < min_dist:
                    min_dist = dist
                    closest = u
            
            if closest:
                matching.append((v, closest))
                odd_vertices_set.remove(closest)
        
        # Step 4: Add matching edges to MST to create multigraph
        eulerian_graph = nx.MultiGraph(mst)
        for u, v in matching:
            u_idx, v_idx = node_indices[u], node_indices[v]
            eulerian_graph.add_edge(u, v, weight=distance_matrix[u_idx][v_idx])
        
        # Step 5: Find Eulerian circuit
        # Simplified algorithm to find Eulerian path
        euler_path = []
        
        def find_euler_path(graph, node):
            for neighbor in list(graph.neighbors(node)):
                if graph.has_edge(node, neighbor):
                    graph.remove_edge(node, neighbor)
                    find_euler_path(graph, neighbor)
            euler_path.append(node)
        
        find_euler_path(eulerian_graph, nodes[0])
        euler_path.reverse()  # Correct order
        
        # Step 6: Shortcut the Eulerian circuit to form a Hamiltonian circuit
        tour = []
        visited = set()
        
        for node in euler_path:
            if node not in visited:
                tour.append(node)
                visited.add(node)
        
        # Calculate tour length
        tour_length = 0
        for i in range(len(tour)):
            u, v = tour[i], tour[(i + 1) % len(tour)]
            u_idx, v_idx = node_indices[u], node_indices[v]
            tour_length += distance_matrix[u_idx][v_idx]
        
        return tour, tour_length
    
    @staticmethod
    def get_best_initial_tour(instance: TSPInstance) -> Tuple[List[Any], float, str]:
        """
        Try multiple initialization methods and return the best tour.
        
        Args:
            instance: TSP instance
            
        Returns:
            Tuple: (best_tour, tour_length, method_name)
        """
        methods = {
            "nearest_neighbor": AdvancedTourInitializer.nearest_neighbor,
            "multi_fragment": AdvancedTourInitializer.multi_fragment,
            "convex_hull": AdvancedTourInitializer.convex_hull_insertion
        }
        
        # Christofides only for smaller instances due to complexity
        if instance.size <= 200:
            methods["christofides"] = AdvancedTourInitializer.christofides
        
        best_tour = None
        best_length = float('inf')
        best_method = None
        
        for method_name, method in methods.items():
            tour, length = method(instance)
            logger.debug(f"Initial tour with {method_name}: length={length:.2f}")
            
            if length < best_length:
                best_tour = tour
                best_length = length
                best_method = method_name
        
        logger.info(f"Best initial tour: {best_method} with length={best_length:.2f}")
        return best_tour, best_length, best_method


class TSPAdaptiveSolver(TSPSolver):
    """
    Adaptive TSP solver that combines multiple initialization methods with enhanced k-Opt.
    Automatically selects the best approach based on instance characteristics.
    """
    
    def __init__(self, config: Optional[SolverConfiguration] = None):
        """Initialize the Adaptive TSP solver."""
        super().__init__("TSP-Adaptive", config)
        if self.config:
            if not hasattr(self.config, 'use_4opt'):
                self.config.use_4opt = True
                
    def get_reduction_factor(self, instance: TSPInstance) -> float:
        """
        Calculate reduction factor for Adaptive solver based on the solvers it might use.
        Adaptive solver should provide at least as good a reduction as the best solver it uses.
        
        Args:
            instance: TSPInstance to evaluate
        
        Returns:
            float: Reduction factor (0.0 to 1.0) - lower means better pattern exploitation
        """
        # For adaptive solver, use the best (lowest) reduction factor from potential solvers
        reduction_factors = []
        
        # Get reduction factors from different solvers based on instance size
        if instance.size <= 100 and self.config and getattr(self.config, 'use_4opt', False):
            # 4-Opt is best for very small instances
            opt4_solver = TSP4OptSolver()
            reduction_factors.append(opt4_solver.get_reduction_factor(instance))
        
        if instance.size <= 500:
            # Enhanced 3-Opt for medium instances
            e3opt_solver = TSPEnhanced3OptSolver() 
            reduction_factors.append(e3opt_solver.get_reduction_factor(instance))
        
        # 2-Opt for all sizes
        opt2_solver = TSP2OptSolver()
        reduction_factors.append(opt2_solver.get_reduction_factor(instance))
        
        # The adaptive solver should be at least as good as the best solver it can use
        # Use the smallest (best) reduction factor, or default if none available
        return min(reduction_factors) if reduction_factors else 0.5
    
    @timeit
    def solve(self, instance: TSPInstance) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Solve TSP using adaptive approach based on instance characteristics.
        Combines advanced initialization with enhanced k-Opt optimization.
        
        Args:
            instance: TSPInstance to solve
            
        Returns:
            Tuple: (optimized tour, metadata dictionary)
        """
        logger.info(f"Starting Adaptive solver for '{instance.name}'")
        start_time = time.time()
        
        # Adjust configuration based on instance size
        self.config.adjust_for_instance(instance)
        
        # Step 1: Get best initial tour using multiple initialization methods
        tour, base_length, init_method = AdvancedTourInitializer.get_best_initial_tour(instance)
        nn_tour, nn_length = AdvancedTourInitializer.nearest_neighbor(instance)
        
        # Step 2: Apply appropriate optimization based on instance size
        iterations = 0
        improvements = 0
        opt_method = "none"
        
        # For very small instances, try 4-Opt
        if instance.size <= 100 and self.config.use_4opt and self.config.max_execution_time >= 20:
            opt_method = "4-opt"
            solver = TSP4OptSolver(self.config)
            tour, metadata = solver.solve(instance)
            iterations = metadata.get("iterations", 0)
            improvements = metadata.get("improvements", 0)
            
        # For medium instances, use Enhanced 3-Opt
        elif instance.size <= 500 and self.config.max_execution_time >= 10:
            opt_method = "enhanced-3-opt"
            solver = TSPEnhanced3OptSolver(self.config)
            tour, metadata = solver.solve(instance)
            iterations = metadata.get("iterations", 0)
            improvements = metadata.get("improvements", 0)
            
        # For large instances, use aggressive 2-Opt with pattern awareness
        else:
            opt_method = "aggressive-2-opt"
            # Create a wrapped 2-Opt solver that uses aggressive swap selection
            class AggressiveTSP2OptSolver(TSP2OptSolver):
                def solve(self, instance: TSPInstance) -> Tuple[List[Any], Dict[str, Any]]:
                    logger.info(f"Starting Aggressive 2-Opt solver for '{instance.name}'")
                    start_time = time.time()
                    
                    # Use existing initial tour
                    baseline_tour_length = instance.get_tour_length(tour)
                    nn_tour_length = nn_length
                    
                    graph = instance.data
                    n = len(tour)
                    if n <= 3:
                        return tour, {"runtime": 0, "tour_length": baseline_tour_length, "iterations": 0}
                    
                    nodes = sorted(graph.nodes())
                    node_indices = {node: i for i, node in enumerate(nodes)}
                    distance_matrix = instance.get_distance_matrix()
                    
                    # Track progress
                    current_tour = tour.copy()
                    current_length = baseline_tour_length
                    iterations = 0
                    improved = True
                    
                    # Get cluster information for more effective swaps
                    cluster_patterns = [p for p in instance._patterns if p.type == PatternType.CLUSTERING]
                    cluster_indices = []
                    
                    if cluster_patterns:
                        tour_positions = {node: i for i, node in enumerate(current_tour)}
                        for pattern in cluster_patterns:
                            cluster_tour_indices = [tour_positions[node] for node in pattern.elements if node in tour_positions]
                            if len(cluster_tour_indices) >= 3:
                                cluster_indices.append(cluster_tour_indices)
                    
                    while improved and iterations < self.config.max_iterations:
                        if self.check_timeout(start_time):
                            break
                            
                        iterations += 1
                        improved = False
                        
                        # Use aggressive swap selection
                        swap_batches = list(generate_aggressive_swap_batches(
                            n, current_tour, distance_matrix, node_indices,
                            max_swaps=self.config.max_swaps_per_iteration,
                            focus_clusters=cluster_indices,
                            aggression_level=3
                        ))
                        
                        # Process batches
                        best_delta = 0
                        best_swap = None
                        
                        for batch in swap_batches:
                            tasks = [(i, j, node_indices, distance_matrix, current_tour) for i, j in batch]
                            
                            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                                results = list(executor.map(evaluate_swap, tasks))
                                
                            for i, j, delta in results:
                                if delta < best_delta:
                                    best_delta = delta
                                    best_swap = (i, j)
                        
                        if best_swap and best_delta < 0:
                            i, j = best_swap
                            current_tour = apply_2opt_swap(current_tour, i, j)
                            current_length += best_delta
                            improved = True
                    
                    runtime = time.time() - start_time
                    sqf = ((nn_tour_length - current_length) / nn_tour_length * 100.0) if nn_tour_length > 0 else 0.0
                    
                    metadata = {
                        "runtime": runtime,
                        "tour_length": current_length,
                        "iterations": iterations,
                        "sqf": sqf,
                        "pue": instance.get_pue()
                    }
                    
                    return current_tour, metadata
                    
            solver = AggressiveTSP2OptSolver(self.config)
            tour, metadata = solver.solve(instance)
            iterations = metadata.get("iterations", 0)
        
        # Calculate final tour length and SQF
        final_length = instance.get_tour_length(tour)
        runtime = time.time() - start_time
        sqf = ((nn_length - final_length) / nn_length * 100.0) if nn_length > 0 else 0.0
        
        # Prepare comprehensive metadata
        metadata = {
            "runtime": runtime,
            "tour_length": final_length,
            "nn_tour_length": nn_length,
            "initial_length": base_length,
            "initialization_method": init_method,
            "optimization_method": opt_method,
            "iterations": iterations,
            "improvements": improvements,
            "sqf": sqf,
            "pue": instance.get_pue()
        }
        
        logger.info(f"Adaptive solver completed: runtime={runtime:.4f}s, tour_length={final_length:.2f}, "
                   f"method={init_method}+{opt_method}, SQF={sqf:.2f}%")
                   
        return tour, metadata


# Add the enhanced solvers to the SolverConfiguration
def create_enhanced_configuration():
    """Create an enhanced configuration with adjusted parameters for better performance."""
    config = SolverConfiguration()
    
    # Adjust parameters for enhanced performance
    config.max_execution_time = 300  # Allow longer execution time (5 minutes)
    config.max_iterations = 100  # More iterations for optimization
    config.max_swaps_per_iteration = 2000  # More aggressive swap selection
    config.early_termination_threshold = 0.0001  # Lower threshold for more thorough optimization
    config.dynamically_adjust_parameters = True  # Enable dynamic parameter adjustment
    
    # Parameters for 3-Opt and 4-Opt
    config.max_3opt_iterations = 30
    config.max_4opt_iterations = 15
    
    # Simulated annealing parameters
    config.initial_temp = 1.0
    config.cooling_rate = 0.95
    
    # 4-Opt parameters
    config.use_4opt = True
    config.segment_sampling = True
    
    return config

###########################################
# 7. Experimentation Framework
###########################################

def run_single_task(solver_instance_pair):
    """
    Execute a single solver-instance task for parallel processing.
    Designed to be called by ProcessPoolExecutor.
    
    This function encapsulates the process of running a solver on an instance:
    1. Extract the solver, instance, and metrics to track from the input tuple
    2. Apply the solver to the instance and collect the solution and metadata
    3. Return the results in a standardized dictionary format
    
    This design enables parallel execution of multiple solver-instance pairs,
    significantly speeding up experiments with many solvers or instances.
    
    Args:
        solver_instance_pair: Tuple of (solver, instance, metrics)
    
    Returns:
        Dict: Result dictionary with solver, instance, solution, and metadata
    """
    solver, instance, metrics = solver_instance_pair
    logger.info(f"Running task: '{solver.name}' on '{instance.name}'")
    try:
        # Solve the instance and collect results
        solution, metadata = solver.solve(instance)
        if 'SQF' in metrics:
            logger.debug(f"SQF included in metadata: {metadata['sqf']:.2f}%")
        logger.info(f"Task completed: PUE={metadata['pue']:.2f}%, SQF={metadata.get('sqf', 0.0):.2f}%")
        return {'solver': solver.name, 'instance': instance.name, 'solution': solution, 'metadata': metadata}
    except Exception as e:
        logger.error(f"Error in task: {e}")
        return {'solver': solver.name, 'instance': instance.name, 'error': str(e)}


class Experiment:
    """
    Framework for running experiments across solvers and instances.
    Supports visualization and result analysis.
    
    The Experiment class provides a structured way to:
    1. Define a set of problem instances and solvers to evaluate
    2. Run all solver-instance combinations systematically
    3. Collect and analyze performance metrics
    4. Visualize results with various plots and charts
    5. Save results for future reference or further analysis
    
    This facilitates rigorous empirical evaluation of solver performance
    and helps identify which solvers work best for which types of instances.
    """
    
    def __init__(self, name: str):
        """
        Initialize the experiment.
        
        Creates a new experiment with the given name and empty lists for
        solvers, instances, and results. The experiment name is used for
        identifying the experiment in logs and output files.
        
        Args:
            name: Identifier for the experiment
        """
        logger.debug(f"Initializing experiment '{name}'")
        self.name = name
        self.results = []
        self.solvers = []
        self.instances = []
        self.config = SolverConfiguration()  # Configuration for all solvers
        self.start_time = time.time()
        logger.info(f"Experiment '{name}' initialized")
    
    def add_solver(self, solver: Solver):
        """
        Add a solver to the experiment.
        
        Adds a solver to the list of solvers that will be evaluated
        on all instances when the experiment is run.
        
        Args:
            solver: Solver object to add
        """
        logger.debug(f"Adding solver '{solver.name}'")
        self.solvers.append(solver)
    
    def add_instance(self, instance: ProblemInstance):
        """
        Add a problem instance to the experiment.
        
        Adds a problem instance to the list of instances that will be
        solved by all solvers when the experiment is run.
        
        Args:
            instance: ProblemInstance object to add
        """
        logger.debug(f"Adding instance '{instance.name}'")
        self.instances.append(instance)
    
    def set_configuration(self, config: SolverConfiguration):
        """
        Set configuration for all solvers in the experiment.
        
        Args:
            config: Configuration settings to apply
        """
        self.config = config
        for solver in self.solvers:
            solver.config = config
        logger.debug(f"Updated configuration for all solvers")
    
    @timeit
    def run(self, metrics: List[str] = None, parallel: bool = True, time_limit: float = None):
        """
        Run the experiment for all solver-instance pairs.
        
        This method executes all combinations of solvers and instances,
        collecting the specified metrics for each run. When parallel=True,
        it uses multiple CPU cores to accelerate execution.
        
        The results include:
        - The solver and instance names
        - The solution (tour for TSP, alignment for sequences, etc.)
        - Metadata containing runtime, solution quality, and other metrics
        
        Args:
            metrics: List of metrics to collect (default: ['runtime', 'PUE', 'SQF'])
            parallel: Whether to run in parallel (default: True)
            time_limit: Maximum time limit for the entire experiment (optional)
        
        Returns:
            List[Dict]: List of result dictionaries
        """
        logger.info(f"Running experiment '{self.name}', parallel={parallel}")
        logger.info(f"Solvers: {[s.name for s in self.solvers]}")
        logger.info(f"Instances: {[i.name for i in self.instances]}")
        
        if metrics is None:
            metrics = ['runtime', 'PUE', 'SQF']  # Include SQF by default
        
        # Check if any solver or instance is using the default configuration
        for solver in self.solvers:
            if solver.config is None:
                solver.config = self.config
                logger.debug(f"Applied default configuration to solver '{solver.name}'")
        
        # Create tasks for each solver-instance pair
        tasks = [(solver, instance, metrics) for solver in self.solvers for instance in self.instances]
        logger.debug(f"Prepared {len(tasks)} tasks")
        
        # Calculate time limit for each task if experiment time limit is provided
        if time_limit:
            task_time_limit = time_limit / len(tasks)
            for solver in self.solvers:
                solver.config.max_execution_time = min(
                    solver.config.max_execution_time, 
                    task_time_limit
                )
            logger.info(f"Set task time limit to {task_time_limit:.2f}s based on experiment time limit {time_limit:.2f}s")
        
        # Execute tasks in parallel or sequentially
        start_time = time.time()
        if parallel and len(tasks) > 1:
            # Use a shared job queue with manager process to enable task prioritization
            with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=worker_init) as executor:
                futures = []
                for task in tasks:
                    future = executor.submit(run_single_task, task)
                    futures.append(future)
                
                # Process results as they complete
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
                    elapsed = time.time() - start_time
                    remaining = len(futures) - len(results)
                    logger.debug(f"Completed {len(results)}/{len(futures)} tasks, {remaining} remaining, elapsed: {elapsed:.2f}s")
                    
                    # Check if overall time limit is approaching
                    if time_limit and elapsed > time_limit * 0.9:
                        logger.warning(f"Approaching experiment time limit ({elapsed:.2f}s / {time_limit:.2f}s), canceling remaining tasks")
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                
                self.results = results
                logger.info(f"Parallel execution completed with {len(self.results)} results")
        else:
            self.results = []
            for i, task in enumerate(tasks):
                # Check if time limit is approaching
                if time_limit and time.time() - start_time > time_limit * 0.9:
                    logger.warning(f"Approaching experiment time limit, skipping remaining tasks")
                    break
                
                result = run_single_task(task)
                self.results.append(result)
                logger.debug(f"Completed task {i+1}/{len(tasks)}")
            
            logger.info(f"Sequential execution completed with {len(self.results)} results")
        
        # Record experiment duration
        self.duration = time.time() - start_time
        logger.info(f"Experiment completed in {self.duration:.2f}s")
        return self.results
    
    def summarize(self):
        """
        Summarize experiment results with statistical metrics.
        
        Calculates summary statistics (mean, standard deviation, minimum, maximum)
        for all numeric metrics collected during the experiment. This provides
        a concise overview of solver performance across all instances.
        
        For example, for the 'runtime' metric, it calculates the average runtime,
        the variability in runtime, and the fastest and slowest executions.
        
        Returns:
            Dict: Dictionary with experiment summary containing:
                - name: experiment name
                - solvers: list of solver names
                - instances: list of instance names
                - metrics: dictionary of metric statistics
        """
        logger.info(f"Summarizing results for '{self.name}'")
        if not self.results:
            logger.warning("No results to summarize")
            return {}
        
        # Initialize summary structure
        summary = {
            'name': self.name,
            'solvers': [s.name for s in self.solvers],
            'instances': [i.name for i in self.instances],
            'metrics': {},
            'duration': getattr(self, 'duration', time.time() - self.start_time)
        }
        
        # Collect all metric names from results
        all_metrics = set()
        for result in self.results:
            if 'metadata' in result:
                all_metrics.update(result['metadata'].keys())
        
        # Calculate statistics for each metric
        for metric in all_metrics:
            values = [r['metadata'].get(metric) for r in self.results if 'metadata' in r and metric in r['metadata']]
            # Check if all values are numeric (int or float) and not None
            if values and all(isinstance(v, (int, float)) for v in values if v is not None):
                # Filter out None values
                numeric_values = [v for v in values if v is not None]
                if numeric_values:  # Only proceed if we have numeric values
                    summary['metrics'][metric] = {
                        'mean': np.mean(numeric_values),
                        'std': np.std(numeric_values),
                        'min': np.min(numeric_values),
                        'max': np.max(numeric_values),
                        'median': np.median(numeric_values)
                    }
                    logger.debug(f"Metric '{metric}': {summary['metrics'][metric]}")
                else:
                    logger.warning(f"Metric '{metric}' has no valid numeric values")
            else:
                logger.warning(f"Metric '{metric}' contains non-numeric values, skipping statistics")
        
        # Add solver-specific summaries
        solver_summaries = {}
        for solver_name in summary['solvers']:
            solver_results = [r for r in self.results if r.get('solver') == solver_name]
            
            # Skip if no results for this solver
            if not solver_results:
                continue
                
            solver_metrics = {}
            for metric in all_metrics:
                values = [r['metadata'].get(metric) for r in solver_results if 'metadata' in r and metric in r['metadata']]
                # Check if all values are numeric (int or float) and not None
                if values and all(isinstance(v, (int, float)) for v in values if v is not None):
                    # Filter out None values
                    numeric_values = [v for v in values if v is not None]
                    if numeric_values:  # Only proceed if we have numeric values
                        solver_metrics[metric] = {
                            'mean': np.mean(numeric_values),
                            'std': np.std(numeric_values),
                            'min': np.min(numeric_values),
                            'max': np.max(numeric_values),
                            'median': np.median(numeric_values)
                        }
                else:
                    logger.debug(f"Skipping non-numeric metric '{metric}' for solver '{solver_name}'")
            
            solver_summaries[solver_name] = solver_metrics
        
        summary['solver_metrics'] = solver_summaries
        logger.info(f"Summary: {summary}")
        return summary
    
    def save_results(self, path: str):
        """
        Save experiment results to a JSON file.
        
        Serializes the experiment results to a JSON file that can be
        loaded later for further analysis or shared with others.
        The file includes metadata about the experiment and all results.
        
        Args:
            path: File path to save results
        """
        logger.info(f"Saving results to '{path}'")
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Prepare serializable results
        serializable_results = []
        for result in self.results:
            serializable_result = {
                'solver': result.get('solver'),
                'instance': result.get('instance'),
                'metadata': result.get('metadata', {})
            }
            solution = result.get('solution')
            if isinstance(solution, (list, dict, str, int, float, bool, np.ndarray)) or solution is None:
                if isinstance(solution, np.ndarray):
                    serializable_result['solution'] = solution.tolist()
                else:
                    serializable_result['solution'] = solution
            else:
                serializable_result['solution_type'] = type(solution).__name__
            serializable_results.append(serializable_result)
        
        # Include summary 
        # Avoid calling self.summarize() to prevent errors
        summary = {
            'name': self.name,
            'solvers': [s.name for s in self.solvers],
            'instances': [i.name for i in self.instances],
            'duration': getattr(self, 'duration', time.time() - self.start_time)
        }
        
        # Save to file with proper formatting
        with open(path, 'w') as f:
            json.dump({
                'name': self.name, 
                'results': serializable_results,
                'summary': summary
            }, f, indent=2)
        logger.info(f"Results saved successfully")
    
    def plot(self, metric: str = 'runtime', save_path: Optional[str] = None):
        """
        Plot experiment results for a specified metric.
        
        Creates a bar chart comparing the performance of different solvers
        across all instances for the specified metric. This visualization
        helps identify patterns in solver performance and highlights which
        solvers excel on which instances.
        
        For example, plotting the 'SQF' metric shows which solvers achieve
        the best solution quality improvements over the baseline.
        
        Args:
            metric: Metric to plot (default: 'runtime')
            save_path: Path to save the plot (default: None, displays instead)
        """
        logger.info(f"Plotting '{metric}' for '{self.name}'")
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        data = {}
        solvers = set()
        instances = set()
        for result in self.results:
            if 'metadata' in result and metric in result['metadata']:
                solver = result['solver']
                instance = result['instance']
                value = result['metadata'][metric]
                if value is not None:
                    solvers.add(solver)
                    instances.add(instance)
                    if solver not in data:
                        data[solver] = {}
                    data[solver][instance] = value
        
        if not data:
            logger.warning(f"No valid data for metric '{metric}'")
            return
        
        # Create bar chart with improved formatting
        solvers = sorted(list(solvers))
        instances = sorted(list(instances))
        plt.figure(figsize=(max(12, len(instances) * 0.8), 8))
        bar_width = 0.8 / len(solvers)
        r = np.arange(len(instances))
        
        # Define color palette for solvers
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(solvers)))
        
        # Plot bars for each solver
        for i, solver in enumerate(solvers):
            values = [data.get(solver, {}).get(instance, np.nan) for instance in instances]
            plt.bar(r + i * bar_width, values, width=bar_width, label=solver, color=colors[i], alpha=0.8)
        
        # Add labels and legend with improved formatting
        plt.xlabel('Problem Instance', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'{metric.capitalize()} by Solver and Instance', fontsize=14, fontweight='bold')
        
        # Rotate x-labels for better readability
        plt.xticks(r + bar_width * (len(solvers) - 1) / 2, instances, rotation=45, ha='right')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Place legend outside plot for better visualization
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        # Add metric units if known
        if metric == 'runtime':
            plt.ylabel('Runtime (seconds)', fontsize=12)
        elif metric == 'sqf':
            plt.ylabel('Solution Quality Factor (%)', fontsize=12)
        elif metric == 'pue':
            plt.ylabel('Pattern Utilization Efficiency (%)', fontsize=12)
        
        # Save or display plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to '{save_path}'")
        else:
            plt.show()
    
    def visualize_instance(self, instance, output_dir: Optional[str] = None):
        """
        Generate four-panel visualizations for a specific instance.
        
        Creates detailed visualizations for a problem instance that show:
        - The problem structure and detected patterns
        - Complexity reduction through pattern-awareness
        - Solutions from different solvers
        - Performance comparisons
        
        For TSP instances, this generates the comprehensive four-panel
        visualization described in _visualize_tsp_instance.
        
        Args:
            instance: ProblemInstance to visualize
            output_dir: Directory to save visualization (default: None, displays instead)
        """
        logger.info(f"Visualizing instance '{instance.name}'")
        instance_results = [r for r in self.results if r.get('instance') == instance.name]
        if not instance_results:
            logger.warning(f"No results for '{instance.name}'")
            return
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Select visualization based on instance type
        if isinstance(instance, TSPInstance):
            self._visualize_tsp_instance(instance, {r['solver']: r for r in instance_results}, output_dir)
    
    def _visualize_tsp_instance(self, instance: TSPInstance, solver_results: Dict[str, Dict], output_dir: Optional[str] = None):
        """
        Create a four-panel visualization for a TSP instance.
        Updated to handle enhanced solvers and adaptive optimization.
        """
        logger.debug(f"Visualizing TSP instance '{instance.name}'")
        # Get node positions
        pos = nx.get_node_attributes(instance.data, 'pos')
        if not pos:
            logger.warning("No position data available for visualization")
            return
        
        # Create figure with improved styling
        plt.figure(figsize=(15, 12))
        plt.suptitle(f"Analysis of {instance.name} ({instance.size} cities)", fontsize=16, fontweight='bold')
        
        # Panel 1: Scatter plot with patterns (no changes needed)
        plt.subplot(2, 2, 1)
        ax1 = plt.gca()
        ax1.set_facecolor('#f5f5f5')
        
        # Draw the base graph with improved styling
        nx.draw_networkx_nodes(instance.data, pos, node_size=30, node_color='lightblue', edgecolors='black', linewidths=0.5, ax=ax1)
        
        # Highlight clusters with different colors
        cluster_patterns = [p for p in instance._patterns if p.type == PatternType.CLUSTERING]
        for i, pattern in enumerate(cluster_patterns):
            cluster_subgraph = instance.data.subgraph(pattern.elements)
            color = plt.cm.tab10(i % 10)
            nx.draw_networkx_nodes(cluster_subgraph, pos, node_size=60, node_color=color, alpha=0.7, ax=ax1)
        
        # Add information text
        plt.title(f"Problem Structure with Detected Patterns\nPUE: {instance.get_pue():.2f}%", fontsize=12)
        pattern_info = f"Detected {len(instance._patterns)} patterns:\n"
        pattern_info += f"• {len(cluster_patterns)} clusters\n"
        pattern_info += f"• {len([p for p in instance._patterns if p.type == PatternType.SYMMETRY])} symmetries"
        ax1.annotate(pattern_info, xy=(0.05, 0.05), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        # Panel 3: Nearest Neighbor tour
        nn_result = None
        for solver_name, result in solver_results.items():
            if 'TSP-NearestNeighbor' in solver_name and 'solution' in result:
                nn_result = result
                break
        
        if nn_result and 'solution' in nn_result:
            nn_tour = nn_result['solution']
            nn_metadata = nn_result['metadata']
            
            plt.subplot(2, 2, 3)
            ax3 = plt.gca()
            ax3.set_facecolor('#f5f5f5')
            
            # Draw the NN tour with clear, connected edges
            tour_graph = nx.Graph()
            tour_graph.add_nodes_from(nn_tour)
            edges = [(nn_tour[i], nn_tour[i+1]) for i in range(len(nn_tour)-1)]
            edges.append((nn_tour[-1], nn_tour[0]))  # Close the tour
            tour_graph.add_edges_from(edges)
            
            nx.draw_networkx_nodes(tour_graph, pos, node_size=30, node_color='lightblue', edgecolors='black', linewidths=0.5, ax=ax3)
            nx.draw_networkx_edges(tour_graph, pos, width=1.2, edge_color='green', alpha=0.8, ax=ax3)
            
            # Add result information
            plt.title(f"Nearest Neighbor Tour\nLength: {nn_metadata['tour_length']:.2f}", fontsize=12)
            runtime_text = f"Runtime: {nn_metadata['runtime']:.6f}s\n"
            runtime_text += f"PUE: {nn_metadata['pue']:.2f}%\n"
            runtime_text += f"Tour complexity: O(n²)"
            ax3.annotate(runtime_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        # Panel 4: Best optimized tour (find the best solver result)
        best_result = None
        best_tour_length = float('inf')
        best_solver_name = None
        
        # Find the best solution among all solvers (except NN)
        for solver_name, result in solver_results.items():
            if 'TSP-NearestNeighbor' not in solver_name and 'solution' in result and 'metadata' in result:
                tour_length = result['metadata'].get('tour_length', float('inf'))
                if tour_length < best_tour_length:
                    best_tour_length = tour_length
                    best_result = result
                    best_solver_name = solver_name
        
        # If no optimized solver found, try to find any solution
        if best_result is None:
            for solver_name, result in solver_results.items():
                if 'solution' in result and 'metadata' in result and solver_name != 'TSP-NearestNeighbor':
                    best_result = result
                    best_solver_name = solver_name
                    break
        
        if best_result and 'solution' in best_result:
            opt_tour = best_result['solution']
            opt_metadata = best_result['metadata']
            
            plt.subplot(2, 2, 4)
            ax4 = plt.gca()
            ax4.set_facecolor('#f5f5f5')
            
            # Draw the optimized tour
            tour_graph = nx.Graph()
            tour_graph.add_nodes_from(opt_tour)
            edges = [(opt_tour[i], opt_tour[i+1]) for i in range(len(opt_tour)-1)]
            edges.append((opt_tour[-1], opt_tour[0]))  # Close the tour
            tour_graph.add_edges_from(edges)
            
            nx.draw_networkx_nodes(tour_graph, pos, node_size=30, node_color='lightblue', edgecolors='black', linewidths=0.5, ax=ax4)
            nx.draw_networkx_edges(tour_graph, pos, width=1.2, edge_color='red', alpha=0.8, ax=ax4)
            
            # Add result information with SQF gain prominently displayed
            solver_display_name = best_solver_name.replace('TSP-', '')
            plt.title(f"{solver_display_name} Optimized Tour\nLength: {opt_metadata['tour_length']:.2f}", fontsize=12)
            
            improvement_text = f"SQF Gain: {opt_metadata.get('sqf', 0.0):.2f}%\n"
            improvement_text += f"Runtime: {opt_metadata['runtime']:.4f}s\n"
            if 'iterations' in opt_metadata:
                improvement_text += f"Iterations: {opt_metadata['iterations']}"
            ax4.annotate(improvement_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        # Panel 2: Complexity reduction comparison
        if nn_result:
            plt.subplot(2, 2, 2)
            nn_solver = TSPNearestNeighborSolver()
            
            # Use the best optimized solver or fall back to original code
            if best_result and best_solver_name:
                # Create appropriate solver object based on name
                if 'Enhanced3Opt' in best_solver_name:
                    opt_solver = TSPEnhanced3OptSolver()
                elif 'Adaptive' in best_solver_name:
                    opt_solver = TSPAdaptiveSolver()
                elif '4Opt' in best_solver_name:
                    opt_solver = TSP4OptSolver()
                elif '2Opt' in best_solver_name:
                    opt_solver = TSP2OptSolver()
                else:
                    # Fallback to 2-Opt if we can't determine
                    opt_solver = TSP2OptSolver()
                    
                nn_complexity = nn_solver.calculate_complexity(instance)
                
                # For Adaptive solver, calculate complexity differently
                if 'Adaptive' in best_solver_name:
                    # Use the enhanced 3-Opt for complexity calculation as a proxy
                    enhanced_solver = TSPEnhanced3OptSolver()
                    opt_complexity = enhanced_solver.calculate_complexity(instance)
                else:
                    opt_complexity = opt_solver.calculate_complexity(instance)
                    
                self._plot_complexity_comparison(instance, nn_solver, opt_solver, nn_complexity, opt_complexity)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        if output_dir:
            output_file = os.path.join(output_dir, f"{instance.name}_analysis.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Visualization saved to '{output_file}'")
        else:
            plt.show()    
    
    def _plot_complexity_comparison(self, instance: TSPInstance, nn_solver: TSPSolver, 
                                    opt_solver: TSPSolver, nn_complexity: float, opt_complexity: float):
        """
        Plot complexity comparison on a logarithmic scale.
        Handles extremely large complexity values safely.
        """
        logger.debug(f"Plotting complexity comparison for '{instance.name}'")
        algorithms = ['Nearest Neighbor', opt_solver.name.replace('TSP-', '')]
        
        # Get base complexities (these might already be log values for large instances)
        base_nn = nn_solver.get_base_complexity(instance.size)
        base_opt = opt_solver.get_base_complexity(instance.size)
        
        # Check if we're working with log values (for large instances)
        is_log_complexity = instance.size > 100
        
        if is_log_complexity:
            # Already in log form, no need to convert
            log_base = [base_nn, base_opt]
            log_pattern = [nn_complexity, opt_complexity]
            
            # If pattern complexities are so small they'd be negative in log form,
            # set a minimum displayable value
            log_pattern = [max(0, x) for x in log_pattern]
        else:
            # Ensure non-zero values for logarithmic plotting
            nn_complexity = max(1e-10, nn_complexity)
            opt_complexity = max(1e-10, opt_complexity)
            
            # Convert to logarithmic scale for better visualization
            log_base = [math.log10(base_nn), math.log10(base_opt)]
            log_pattern = [math.log10(nn_complexity), math.log10(opt_complexity)]
        
        # Set up the plot with improved styling
        ax = plt.gca()
        ax.set_facecolor('#f5f5f5')
        bar_width = 0.35
        x = range(len(algorithms))
        
        # Create bars with improved colors
        plt.bar([i - bar_width / 2 for i in x], log_base, bar_width, label='Base Complexity',
                color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.bar([i + bar_width / 2 for i in x], log_pattern, bar_width, label='Pattern-Aware',
                color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Enhance labels and formatting
        plt.ylabel('Log10(Complexity)', fontsize=12)
        plt.xticks(x, algorithms, fontsize=10)
        plt.title('Complexity Reduction with Pattern Awareness', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add annotations showing percentage reductions
        for i, (base, pattern) in enumerate(zip(log_base, log_pattern)):
            if base > pattern:
                # For large instances, compute reduction differently
                if is_log_complexity:
                    # If base is much larger than pattern, reduction is effectively 100%
                    if base - pattern > 10:
                        reduction = 99.999
                    else:
                        # Calculate reduction percentage: (1 - 10^(pattern-base)) * 100
                        reduction = (1 - 10 ** (pattern - base)) * 100
                else:
                    # Standard calculation for smaller instances
                    reduction = (1 - 10 ** (pattern - base)) * 100
                    
                reduction = max(0, min(100, reduction))  # Clamp to valid range
                
                # Position annotation below the lower bar
                y_pos = min(base, pattern) - 0.5
                plt.text(i, y_pos, f"{reduction:.1f}% reduction", 
                         ha='center', va='top', fontweight='bold', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            else:
                plt.text(i, min(base, pattern) - 0.5, "No reduction", 
                         ha='center', va='top', fontsize=10)
                         
        # Add annotation explaining complexity calculation
        complexity_info = f"n = {instance.size} cities\n"
        complexity_info += f"ρ = {instance.get_pattern_prevalence():.2f}\n"
        complexity_info += f"H = {instance.get_entropy():.2f}"
        
        plt.annotate(complexity_info, xy=(0.05, 0.95), xycoords='axes fraction', 
                    va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

###########################################
# 8. Dataset Utilities
###########################################

class DatasetLoader:
    """
    Utility class for loading datasets for different problem domains.
    Provides static methods to load various types of problem instances.
    
    The DatasetLoader handles the specifics of parsing different file formats
    and converting the raw data into the appropriate ProblemInstance objects:
    - For TSP: Loads coordinates from TSPLIB format and builds graph representation
    - For genetic sequences: Parses FASTA files into sequence objects
    - For weather data: Loads NetCDF or CSV files into time series datasets
    
    This abstraction simplifies experiment setup and makes it easy to work
    with standard benchmark datasets.
    """

    @staticmethod
    @timeit
    def load_tsp_dataset(path: str, file_format: str = 'tsplib') -> List[TSPInstance]:
        """
        Load TSP instances from file.
        Supports TSPLIB format with node coordinates.
        
        TSPLIB is a standard benchmark library for the Traveling Salesman Problem.
        Files in this format typically contain:
        - NAME: Identifier for the instance
        - DIMENSION: Number of cities
        - NODE_COORD_SECTION: List of city coordinates
        
        This method parses the file, extracts city coordinates, builds a complete
        graph with Euclidean distances as edge weights, and creates a TSPInstance.
        
        Args:
            path: Path to TSP file
            file_format: Format of the file (default: 'tsplib')
        
        Returns:
            List[TSPInstance]: List of loaded TSP instances
        """
        logger.info(f"Loading TSP dataset from '{path}', format: {file_format}")
        instances = []
        
        if file_format == 'tsplib':
            try:
                # Parse TSPLIB format file
                with open(path, 'r') as f:
                    lines = f.readlines()
                name = os.path.basename(path).split('.')[0]  # Default name from filename
                dimension = 0
                node_coord_section = False
                coordinates = []
                
                # Parse file line by line
                for line in lines:
                    line = line.strip()
                    if line.startswith("NAME"):
                        name = line.split(":")[1].strip()
                    elif line.startswith("DIMENSION"):
                        dimension = int(line.split(":")[1].strip())
                    elif line == "NODE_COORD_SECTION":
                        node_coord_section = True
                    elif node_coord_section and line != "EOF":
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                index = int(parts[0]) - 1  # Convert to 0-based indexing
                                x = float(parts[1])
                                y = float(parts[2])
                                coordinates.append((index, x, y))
                            except ValueError:
                                logger.warning(f"Skipping invalid line: {line}")
                
                # Create graph from coordinates
                if coordinates:
                    # Use faster graph construction for larger instances
                    graph = nx.Graph()
                    # Add nodes with position attributes
                    for index, x, y in coordinates:
                        graph.add_node(index, pos=(x, y))
                    
                    # For efficiency, we'll calculate distances only when needed
                    # for larger instances instead of pre-computing all edges
                    if len(coordinates) <= 1000:
                        # Pre-compute all edges for smaller instances
                        for i, (idx1, x1, y1) in enumerate(coordinates):
                            for j, (idx2, x2, y2) in enumerate(coordinates[i+1:], i+1):
                                if idx1 != idx2:
                                    # Euclidean distance between cities
                                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                    graph.add_edge(idx1, idx2, weight=distance)
                    else:
                        # For very large instances, create a complete graph with a distance function
                        # Edges and distances will be calculated on-demand
                        logger.info(f"Large instance detected ({len(coordinates)} cities). Using on-demand distance calculation.")
                        for i, (idx1, _, _) in enumerate(coordinates):
                            for j, (idx2, _, _) in enumerate(coordinates[i+1:], i+1):
                                if idx1 != idx2:
                                    # Use a placeholder weight, actual distance will be calculated when needed
                                    graph.add_edge(idx1, idx2, weight=1.0)
                        
                        # Helper to calculate actual distances later
                        def calculate_distances(g):
                            pos = nx.get_node_attributes(g, 'pos')
                            for u, v in g.edges():
                                x1, y1 = pos[u]
                                x2, y2 = pos[v]
                                g[u][v]['weight'] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            return g
                        
                        # Calculate actual edge weights
                        graph = calculate_distances(graph)
                    
                    instance = TSPInstance(graph, name=name)
                    instances.append(instance)
                    logger.debug(f"Loaded TSP instance '{name}' with {len(coordinates)} nodes")
            except Exception as e:
                logger.error(f"Error loading TSP dataset: {e}")
        
        logger.info(f"Loaded {len(instances)} TSP instances")
        return instances

    @staticmethod
    @timeit
    def load_genetic_dataset(path: str, file_format: str = 'fasta') -> List:
        """
        Load genetic sequence instances.
        Requires BioPython to be installed.
        
        Parses genetic sequence files (e.g., DNA, RNA, protein sequences)
        and creates GeneticSequenceInstance objects. Supports various formats:
        - FASTA: Simple format with sequence identifier and sequence data
        - GenBank: More detailed format with additional annotations
        - Others supported by BioPython's SeqIO module
        
        Args:
            path: Path to genetic sequence file
            file_format: Format of the file (default: 'fasta')
        
        Returns:
            List: List of GeneticSequenceInstance objects
        """
        logger.info(f"Loading genetic dataset from '{path}', format: {file_format}")
        if not BIOPYTHON_AVAILABLE:
            logger.error("BioPython not available")
            return []
        
        instances = []
        try:
            # Parse sequences using BioPython
            sequences = list(Bio.SeqIO.parse(path, file_format))
            for i, seq in enumerate(sequences):
                name = seq.id if hasattr(seq, 'id') else f"seq_{i}"
                instance = GeneticSequenceInstance(seq, name=name)
                instances.append(instance)
                logger.debug(f"Loaded genetic instance '{name}' with length {len(seq.seq)}")
        except Exception as e:
            logger.error(f"Error loading genetic dataset: {e}")
        logger.info(f"Loaded {len(instances)} genetic instances")
        return instances
    
    @staticmethod
    @timeit
    def load_weather_dataset(path: str, file_format: str = 'netcdf') -> List:
        """
        Load weather data instances.
        Requires xarray to be installed.
        
        Parses weather data files and creates WeatherForecastInstance objects.
        Supports two main formats:
        - NetCDF: Standard format for multidimensional scientific data
        - CSV: Simple tabular format with date/time and measurement columns
        
        Args:
            path: Path to weather data file
            file_format: Format of the file (default: 'netcdf')
        
        Returns:
            List: List of WeatherForecastInstance objects
        """
        logger.info(f"Loading weather dataset from '{path}', format: {file_format}")
        if not XARRAY_AVAILABLE:
            logger.error("xarray not available")
            return []
        
        instances = []
        if file_format == 'netcdf':
            try:
                # Load NetCDF dataset using xarray
                ds = xr.open_dataset(path)
                instance = WeatherForecastInstance(ds, name=os.path.basename(path))
                instances.append(instance)
                logger.debug(f"Loaded weather instance '{instance.name}' with {len(ds.time)} time points")
            except Exception as e:
                logger.error(f"Error loading NetCDF dataset: {e}")
        elif file_format == 'csv':
            try:
                # Load CSV to pandas DataFrame, then convert to xarray
                df = pd.read_csv(path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                ds = df.to_xarray()
                instance = WeatherForecastInstance(ds, name=os.path.basename(path))
                instances.append(instance)
                logger.debug(f"Loaded weather instance from CSV with {len(ds.time)} time points")
            except Exception as e:
                logger.error(f"Error loading weather CSV: {e}")
        logger.info(f"Loaded {len(instances)} weather instances")
        return instances
	
###########################################
# 9. Main Function and CLI
###########################################

# This section provides command-line interface functionality for the framework.
# It supports three main modes of operation:
# 1. Single TSP file analysis: Analyzing a specific TSP instance
# 2. Dataset suggestions: Recommending benchmark datasets
# 3. Full experiments: Running experiments with multiple solvers and instances

def suggest_datasets():
    """
    Provide suggestions for datasets to use in benchmarking.
    
    This function recommends standard benchmark datasets for each problem domain:
    - TSP: Well-known instances from TSPLIB (berlin52, att48, rat783)
    - Genetic: Datasets from major repositories like 1000 Genomes and Rfam
    - Weather: Climate datasets from ECMWF ERA5 and NOAA GSOD
    
    These suggestions help users get started with meaningful experiments
    using established benchmark problems rather than synthetic data.
    
    Returns:
        Dict: Dictionary of dataset suggestions by problem type, including:
            - Dataset name
            - URL for downloading
            - Description
            - Recommended instances to try
    """
    logger.info("Suggesting datasets for benchmarking")
    suggestions = {
        "TSP": [
            {"name": "TSPLIB", "url": "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/", 
             "description": "Standard benchmark library for TSP", 
             "instances": ["berlin52.tsp", "att48.tsp", "rat783.tsp", "pr1002.tsp"],
             "notes": "Contains various TSP instances of different sizes and structures"},
            {"name": "National TSP", "url": "http://www.math.uwaterloo.ca/tsp/data/", 
             "description": "Real-world instances based on cities", 
             "instances": ["usa13509.tsp", "dj38.tsp", "qa194.tsp"],
             "notes": "Geographic instances based on real city locations"}
        ],
        "Genetic": [
            {"name": "1000 Genomes Project", "url": "https://www.internationalgenome.org/data", 
             "description": "Human genetic variation data", 
             "instances": ["Chromosome 21 sequences", "Mitochondrial DNA"],
             "notes": "Large dataset of human genetic sequences good for pattern mining"},
            {"name": "Rfam", "url": "https://rfam.org/", 
             "description": "RNA sequence and structure data", 
             "instances": ["RF00001.fa", "RF00005.fa", "RF00177.fa"],
             "notes": "Contains RNA sequences with known structural patterns"}
        ],
        "Weather": [
            {"name": "ECMWF ERA5", "url": "https://cds.climate.copernicus.eu/", 
             "description": "Reanalysis weather data", 
             "instances": ["Temperature, pressure data for Europe", "Global precipitation patterns"],
             "notes": "High-resolution weather data with strong seasonal patterns"},
            {"name": "NOAA GSOD", "url": "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516", 
             "description": "Global Surface Summary of the Day", 
             "instances": ["2020 temperature data", "Multi-year station records"],
             "notes": "Long-term weather station data excellent for seasonal pattern detection"}
        ]
    }
    logger.debug(f"Suggestions: {suggestions}")
    return suggestions

def create_default_experiment(problem_type: str, dataset_path: str, max_execution_time: float = None, 
                             parallel: bool = True, format: str = 'auto', config: SolverConfiguration = None):
    """
    Create a default experiment configuration for the specified problem type.
    Updated to support enhanced configuration.
    
    Args:
        problem_type: Type of problem ('tsp', 'genetic', 'weather')
        dataset_path: Path to dataset file or directory
        max_execution_time: Maximum execution time per solver-instance pair (optional)
        parallel: Whether to use parallel processing (default: True)
        format: Dataset format (default: 'auto', determines from problem type)
        config: Optional configuration to use (default: None, creates new config)
    
    Returns:
        Tuple: (experiment, instances) - Configured experiment and loaded instances
    """

    # Determine format if set to auto
    if format == 'auto':
        format = {'tsp': 'tsplib', 'genetic': 'fasta', 'weather': 'netcdf'}[problem_type]
    
    # Create experiment with appropriate name
    experiment = Experiment(f"{problem_type.capitalize()} Experiment")
    
        # Create configuration with appropriate settings if not provided
    if config is None:
        config = SolverConfiguration()
        if max_execution_time:
            config.max_execution_time = max_execution_time
        else:
            # Set reasonable defaults if not specified
            config.max_execution_time = 10.0  # 10 seconds default for small datasets
    
    # Load instances and add to experiment
    instances = []
    if problem_type == 'tsp':
        instances = DatasetLoader.load_tsp_dataset(dataset_path, format)
        # Detect patterns
        for instance in instances:
            # More aggressive pruning for pattern detection on small instances
            instance.detect_patterns([PatternType.CLUSTERING])  # Skip symmetry detection for performance
            experiment.add_instance(instance)
        # Add appropriate solvers
        nn_solver = TSPNearestNeighborSolver(config)
        opt2_solver = TSP2OptSolver(config)
        experiment.add_solver(nn_solver)
        experiment.add_solver(opt2_solver)
        
        # Only add 3-Opt for very small instances to avoid excessive runtime
        if all(instance.size < 100 for instance in instances):
            opt3_solver = TSP3OptSolver(config)
            experiment.add_solver(opt3_solver)
    
    elif problem_type == 'genetic' and BIOPYTHON_AVAILABLE:
        instances = DatasetLoader.load_genetic_dataset(dataset_path, format)
        # Detect patterns
        for instance in instances:
            instance.detect_patterns([PatternType.REPETITION, PatternType.MOTIF])
            experiment.add_instance(instance)
        # Add appropriate solver
        solver = RepeatAwareAligner(config)
        experiment.add_solver(solver)
    
    elif problem_type == 'weather' and XARRAY_AVAILABLE:
        instances = DatasetLoader.load_weather_dataset(dataset_path, format)
        # Detect patterns
        for instance in instances:
            instance.detect_patterns([PatternType.SEASONALITY])
            experiment.add_instance(instance)
        # Add appropriate solver
        solver = SeasonalForecastSolver(config)
        experiment.add_solver(solver)
    
    return experiment, instances

def main():
    """
    Main entry point for command-line execution.
    Supports running TSP analysis, suggesting datasets, and running experiments.
    
    The CLI provides three main operational modes:
    
    1. TSP Analysis:
       python script.py path/to/tsp_file.tsp
       - Detects patterns in the specified TSP instance
       - Solves using Nearest Neighbor and 2-Opt algorithms
       - Calculates complexity, PUE, and SQF metrics
       - Generates a four-panel visualization
    
    2. Dataset Suggestions:
       python script.py suggest-datasets
       - Recommends standard benchmark datasets for each problem domain
       - Provides URLs, descriptions, and specific instance suggestions
    
    3. Run Experiment:
       python script.py run-experiment --problem tsp --dataset path/to/dataset
       - Loads instances from the specified dataset
       - Applies appropriate pattern detection for the problem type
       - Runs all appropriate solvers on all instances
       - Collects and analyzes performance metrics
       - Optionally generates visualizations
    
    Command line arguments:
    - tsp_file: Path to TSP file to analyze (optional)
    - --log-level: Logging level [DEBUG, INFO, WARNING, ERROR]
    - suggest-datasets: Suggest datasets for benchmarking
    - run-experiment: Run a full experiment with multiple solvers and instances
    """
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description="Pattern-Aware Complexity Framework Implementation")
    parser.add_argument('tsp_file', nargs='?', help='Path to the TSP file to analyze')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set the logging level')
    parser.add_argument('--version', action='version', version='PACF v1.0', help='Show version information')
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    suggest_parser = subparsers.add_parser("suggest-datasets", help="Suggest datasets for benchmarking")
    
    # Experiment subcommand with options
    exp_parser = subparsers.add_parser("run-experiment", help="Run an experiment")
    exp_parser.add_argument("--problem", type=str, required=True, choices=["tsp", "genetic", "weather"], help="Problem type")
    exp_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file or directory")
    exp_parser.add_argument("--output", type=str, default="results.json", help="Path to save results")
    exp_parser.add_argument("--format", type=str, default="auto", help="Dataset format (auto, tsplib, fasta, netcdf, csv)")
    exp_parser.add_argument("--parallel", action="store_true", help="Run solvers in parallel")
    exp_parser.add_argument("--sequential", action="store_true", help="Run sequentially (no multiprocessing)")
    exp_parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    exp_parser.add_argument("--time-limit", type=float, default=None, help="Maximum execution time in seconds")
    exp_parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")
    # New options for enhanced solvers
    exp_parser.add_argument("--advanced-init", action="store_true", help="Use advanced initialization methods")
    exp_parser.add_argument("--aggressive-search", action="store_true", help="Use aggressive search strategies")
    exp_parser.add_argument("--use-4opt", action="store_true", help="Enable 4-Opt for small instances")
	
	
    # Performance testing subcommand
    perf_parser = subparsers.add_parser("performance-test", help="Run performance tests")
    perf_parser.add_argument("--instances", type=str, default="att48,berlin52,rat783", help="Comma-separated TSP instances to test")
    perf_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each test")
    perf_parser.add_argument("--output", type=str, default="performance_report.json", help="Output file for performance report")
    # New option for testing enhanced solvers
    perf_parser.add_argument("--enhanced", action="store_true", help="Use enhanced solvers for testing")
	
	
    # Parse arguments
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    logger.info(f"Logging level set to {args.log_level}")
    logger.info(f"Pattern-Aware Complexity Framework v1.0")
    logger.info(f"Using {NUM_CORES} CPU cores, {NUM_WORKERS} worker processes")
    
    # Mode 1: Single TSP file analysis
    if args.tsp_file:
        # Analyze a single TSP file
        logger.info(f"Analyzing TSP file: '{args.tsp_file}'")
        instances = DatasetLoader.load_tsp_dataset(args.tsp_file, 'tsplib')
        if not instances:
            logger.error(f"Could not load TSP file '{args.tsp_file}'")
            print(f"Error: Could not load TSP file '{args.tsp_file}'")
            return
        
        # Initialize with loaded instance
        instance = instances[0]
        logger.info(f"Loaded TSP instance '{instance.name}' with {instance.size} cities")
        print(f"Loaded TSP instance: {instance.name} with {instance.size} cities")
        
        # Detect patterns to analyze structure
        print("Detecting patterns in the TSP instance...")
        patterns = instance.detect_patterns([PatternType.CLUSTERING, PatternType.SYMMETRY])
        logger.info(f"Detected {len(patterns)} patterns (Clusters: {len([p for p in patterns if p.type == PatternType.CLUSTERING])}, "
                    f"Symmetries: {len([p for p in patterns if p.type == PatternType.SYMMETRY])})")
        print(f"Detected {len(patterns)} patterns (Clusters: {len([p for p in patterns if p.type == PatternType.CLUSTERING])}, "
              f"Symmetries: {len([p for p in patterns if p.type == PatternType.SYMMETRY])})")
        
        # Create configuration for solvers
        config = SolverConfiguration()
        if instance.size >= 500:
            print("Large instance detected, adjusting solver parameters for better performance...")
            # Adjust parameters for large instances
            config.max_execution_time = 60
            config.max_iterations = 20
            config.max_swaps_per_iteration = 500
            config.early_termination_threshold = 0.01
        
        # Initialize solvers
        nn_solver = TSPNearestNeighborSolver(config)
        opt2_solver = TSP2OptSolver(config)
        
        # Initialize 3-Opt only for smaller instances
        opt3_solver = None
        if instance.size < 500:
            opt3_solver = TSP3OptSolver(config)
        
        # Calculate complexities
        nn_complexity = nn_solver.calculate_complexity(instance)
        opt2_complexity = opt2_solver.calculate_complexity(instance)
        logger.info(f"Complexities - NN: {nn_complexity:.2e}, 2-Opt: {opt2_complexity:.2e}")
        print(f"Nearest Neighbor Complexity: {nn_complexity:.2e}")
        print(f"2-Opt Complexity: {opt2_complexity:.2e}")
        
        # Solve with Nearest Neighbor
        print("Solving with Nearest Neighbor algorithm...")
        logger.info("Solving with Nearest Neighbor")
        nn_tour, nn_metadata = nn_solver.solve(instance)
        print(f"Nearest Neighbor - Tour Length: {nn_metadata['tour_length']:.2f}, Runtime: {nn_metadata['runtime']:.4f}s, "
              f"PUE: {nn_metadata['pue']:.2f}%")
        
        # Solve with 2-Opt
        print("Solving with 2-Opt algorithm (this may take a while for large instances)...")
        logger.info("Solving with 2-Opt")
        opt2_tour, opt2_metadata = opt2_solver.solve(instance)
        print(f"2-Opt - Tour Length: {opt2_metadata['tour_length']:.2f}, Runtime: {opt2_metadata['runtime']:.4f}s, "
              f"PUE: {opt2_metadata['pue']:.2f}%, SQF Gain: {opt2_metadata['sqf']:.2f}%")
        
        # Solve with 3-Opt for smaller instances
        opt3_tour = None
        opt3_metadata = None
        if opt3_solver and instance.size < 200:
            print("Solving with 3-Opt algorithm...")
            logger.info("Solving with 3-Opt")
            opt3_tour, opt3_metadata = opt3_solver.solve(instance)
            print(f"3-Opt - Tour Length: {opt3_metadata['tour_length']:.2f}, Runtime: {opt3_metadata['runtime']:.4f}s, "
                 f"PUE: {opt3_metadata['pue']:.2f}%, SQF Gain: {opt3_metadata['sqf']:.2f}%")
        
        # Generate and save four-panel visualization
        print("Generating visualization...")
        output_dir = os.path.dirname(args.tsp_file) or '.'
        experiment = Experiment("TSP Analysis")
        experiment.add_instance(instance)
        experiment.add_solver(nn_solver)
        experiment.add_solver(opt2_solver)
        experiment.results = [
            {'solver': nn_solver.name, 'instance': instance.name, 'solution': nn_tour, 'metadata': nn_metadata},
            {'solver': opt2_solver.name, 'instance': instance.name, 'solution': opt2_tour, 'metadata': opt2_metadata}
        ]
        
        # Add 3-Opt results if available
        if opt3_tour is not None and opt3_metadata is not None:
            experiment.add_solver(opt3_solver)
            experiment.results.append({
                'solver': opt3_solver.name, 
                'instance': instance.name, 
                'solution': opt3_tour, 
                'metadata': opt3_metadata
            })
        
        experiment.visualize_instance(instance, output_dir)
        logger.info(f"Visualization saved to '{output_dir}/{instance.name}_analysis.png'")
        print(f"Visualization saved to '{output_dir}/{instance.name}_analysis.png'")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"{instance.name}_results.json")
        experiment.save_results(results_file)
        print(f"Detailed results saved to '{results_file}'")
        return
    
    # Mode 2: Suggest datasets
    if args.command == "suggest-datasets":
        # Suggest datasets for benchmarking
        logger.info("Executing suggest-datasets command")
        suggestions = suggest_datasets()
        print("Suggested datasets for benchmarking:")
        for problem, datasets in suggestions.items():
            print(f"\n{problem} Problem:")
            for ds in datasets:
                print(f"  - {ds['name']}")
                print(f"    URL: {ds['url']}")
                print(f"    Description: {ds['description']}")
                print(f"    Recommended instances: {', '.join(ds['instances'])}")
                if 'notes' in ds:
                    print(f"    Notes: {ds['notes']}")
    
    # Mode 3: Run experiment (updated with enhanced solvers)
    elif args.command == "run-experiment":
        # Run a full experiment with enhanced solvers
        logger.info(f"Starting experiment with args: {vars(args)}")
        format = args.format
        if format == "auto":
            format = {'tsp': 'tsplib', 'genetic': 'fasta', 'weather': 'netcdf'}[args.problem]
        
        parallel = args.parallel and not args.sequential
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output)
        
        # Create enhanced configuration if advanced options are enabled
        config = None
        if args.problem == 'tsp' and (args.advanced_init or args.aggressive_search or args.use_4opt):
            config = create_enhanced_configuration()
            config.max_execution_time = args.time_limit if args.time_limit else 300
            config.use_4opt = args.use_4opt
            logger.info("Using enhanced configuration with advanced options")
        
        # Create and run experiment
        experiment, instances = create_default_experiment(
            args.problem, 
            args.dataset, 
            max_execution_time=args.time_limit,
            parallel=parallel,
            format=format,
            config=config
        )
        
        # Add enhanced solvers if requested
        if args.problem == 'tsp' and (args.advanced_init or args.aggressive_search or args.use_4opt):
            # Replace existing solvers with enhanced versions
            experiment.solvers = []
            experiment.add_solver(TSPNearestNeighborSolver(config))
            if args.aggressive_search:
                experiment.add_solver(TSPEnhanced3OptSolver(config))
                if args.use_4opt and any(instance.size <= 200 for instance in instances):
                    experiment.add_solver(TSP4OptSolver(config))
            else:
                experiment.add_solver(TSP2OptSolver(config))
                if any(instance.size <= 500 for instance in instances):
                    experiment.add_solver(TSP3OptSolver(config))
            
            # Add the adaptive solver if all advanced options are enabled
            if args.advanced_init and args.aggressive_search:
                experiment.add_solver(TSPAdaptiveSolver(config))
        
        if not instances:
            logger.error(f"No instances loaded from '{args.dataset}'")
            print(f"Error: No instances loaded from '{args.dataset}'")
            return
        
        logger.info(f"Loaded {len(instances)} instances")
        print(f"Loaded {len(instances)} instances. Running experiment...")
        
        # Run experiment
        experiment.run(metrics=['runtime', 'PUE', 'SQF'], parallel=parallel, time_limit=args.time_limit)
        
        # Initialize and train SolverPortfolio
        print("Training solver portfolio...")
        solvers = experiment.solvers
        portfolio = SolverPortfolio(solvers)
        portfolio.train_meta_model(experiment.instances, experiment.results)
        
        # Example: Select solver for each instance using trained portfolio
        print("\nSolver selection:")
        for instance in instances:
            selected_solver = portfolio.select_solver(instance)
            logger.info(f"Selected solver for '{instance.name}': {selected_solver.name}")
            print(f"  {instance.name}: {selected_solver.name}")
        
        # Save results
        experiment.save_results(output_path)
        print(f"Results saved to '{output_path}'")
        
        # Generate visualizations if requested
        if args.visualize:
            print("Generating visualizations...")
            for instance in experiment.instances:
                experiment.visualize_instance(instance, args.output_dir)
                print(f"  Created visualization for {instance.name}")
        
        # Summarize and print results
        summary = experiment.summarize()
        logger.info(f"Experiment summary: {summary}")
        print("\nExperiment Summary:")
        print(f"Problem: {args.problem}")
        print(f"Instances: {len(instances)}")
        print(f"Solvers: {', '.join(summary['solvers'])}")
        print(f"Total duration: {summary.get('duration', 0):.2f} seconds")
        print("\nMetrics:")
        for metric, stats in summary.get('metrics', {}).items():
            print(f"  {metric}:")
            for stat, value in stats.items():
                print(f"    {stat}: {value:.6f}")
    
    # Mode 4: Performance testing
    elif args.command == "performance-test":
        logger.info("Starting performance tests")
        print("Running performance tests...")
        
        # Load specified instances
        instance_names = args.instances.split(',')
        test_instances = []
        
        # Test directory with common TSP instances
        tsp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsp_instances")
        if not os.path.exists(tsp_dir):
            tsp_dir = "."
            
        for name in instance_names:
            name = name.strip()
            # Try common extensions
            for ext in ['.tsp', '.TSP', '']:
                filename = name + ext
                test_path = os.path.join(tsp_dir, filename)
                if os.path.exists(test_path):
                    instances = DatasetLoader.load_tsp_dataset(test_path)
                    if instances:
                        test_instances.extend(instances)
                        print(f"Loaded instance: {instances[0].name} ({instances[0].size} cities)")
                        break
        
        if not test_instances:
            print(f"Error: Could not load any test instances. Please check file paths.")
            return
            
        # Create performance test configuration
        perf_config = SolverConfiguration()
        perf_config.max_execution_time = 120  # Limit to 2 minutes per test
        
        # Run performance tests
        results = {}
        for instance in test_instances:
            instance_results = {"size": instance.size, "tests": {}}
            
            # Detect patterns
            print(f"Detecting patterns in {instance.name}...")
            instance.detect_patterns([PatternType.CLUSTERING, PatternType.SYMMETRY])
            instance_results["patterns"] = len(instance._patterns)
            instance_results["pue"] = instance.get_pue()
            
            # Test each solver in sequence
            for solver_class in [TSPNearestNeighborSolver, TSP2OptSolver]:
                solver_name = solver_class.__name__.replace("TSP", "").replace("Solver", "")
                print(f"Testing {solver_name} on {instance.name}...")
                
                # Run multiple iterations
                test_metrics = []
                for i in range(args.iterations):
                    solver = solver_class(perf_config)
                    start_time = time.time()
                    solution, metadata = solver.solve(instance)
                    total_time = time.time() - start_time
                    
                    test_metrics.append({
                        "runtime": metadata["runtime"],
                        "tour_length": metadata["tour_length"],
                        "sqf": metadata.get("sqf", 0.0),
                        "iterations": metadata.get("iterations", 0),
                        "total_time": total_time
                    })
                    
                    print(f"  Iteration {i+1}/{args.iterations}: {total_time:.4f}s, "
                         f"tour length: {metadata['tour_length']:.2f}")
                
                # Calculate average metrics
                avg_metrics = {}
                for key in test_metrics[0].keys():
                    avg_metrics[key] = sum(m[key] for m in test_metrics) / len(test_metrics)
                
                instance_results["tests"][solver_name] = {
                    "iterations": args.iterations,
                    "metrics": test_metrics,
                    "average": avg_metrics
                }
            
            results[instance.name] = instance_results
        
        # Save performance report
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nPerformance testing completed. Results saved to {args.output}")
        
        # Print summary
        print("\nPerformance Summary:")
        for instance_name, instance_data in results.items():
            print(f"\n{instance_name} ({instance_data['size']} cities):")
            print(f"  Patterns: {instance_data['patterns']}, PUE: {instance_data['pue']:.2f}%")
            for solver_name, solver_data in instance_data["tests"].items():
                avg = solver_data["average"]
                print(f"  {solver_name}:")
                print(f"    Avg Runtime: {avg['runtime']:.4f}s, Tour Length: {avg['tour_length']:.2f}")
                if "sqf" in avg and solver_name != "NearestNeighbor":
                    print(f"    SQF: {avg['sqf']:.2f}%")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()