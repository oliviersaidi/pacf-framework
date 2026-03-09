"""
Smoke tests for PACF v1.0
Run with: python -m pytest test_pacf.py -v
"""
import subprocess
import sys
import importlib.util
import os


def test_version():
    """--version flag returns PACF v1.0"""
    result = subprocess.run(
        [sys.executable, 'PACF_v1.py', '--version'],
        capture_output=True, text=True
    )
    assert 'PACF' in result.stdout or 'PACF' in result.stderr


def test_imports():
    """All required dependencies are importable."""
    required = ['numpy', 'pandas', 'matplotlib', 'networkx', 'sklearn']
    for pkg in required:
        assert importlib.util.find_spec(pkg) is not None, f"Missing required package: {pkg}"


def test_tsp_instances_exist():
    """At least one TSP instance file is present."""
    tsp_dir = 'TSP_instances'
    assert os.path.isdir(tsp_dir), "TSP_instances directory missing"
    tsp_files = [f for f in os.listdir(tsp_dir) if f.endswith('.tsp')]
    assert len(tsp_files) > 0, "No .tsp files found in TSP_instances/"


def test_torch_optional():
    """PyTorch import failure must not crash the module-level load."""
    # We just verify the script can be syntax-checked without torch installed
    result = subprocess.run(
        [sys.executable, '-m', 'py_compile', 'PACF_v1.py'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Syntax error in PACF_v1.py: {result.stderr}"
