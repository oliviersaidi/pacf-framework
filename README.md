# PATTERN-AWARE COMPLEXITY FRAMEWORK (PACF) v1.0

The TSP data used in this work is from [TSPLIB - University of Heidelberg](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), which is freely available for research purposes.

## OVERVIEW

The **Pattern-Aware Complexity Framework (PACF)** is a Python-based framework designed to analyze and exploit structural patterns in NP-hard optimization problems, such as the Traveling Salesman Problem (TSP), genetic sequence alignment, and weather forecasting. By identifying patterns like symmetry, clustering, and repetition, PACF reduces computational complexity and enhances solver performance. This implementation focuses on TSP but is extensible to other domains with optional dependencies.

Developed and tested on a **MacBook Pro M3 Max running macOS 15.3.1**, the framework should work across platforms (macOS, Linux, Windows) with the appropriate setup.

## DIRECTORY STRUCTURE

The `pacf_v1.0/` directory contains all essential files and folders for running the framework:

pacf_v1.0/
├── PACF_v1.py         # Main script
├── tsp_instances/     # Directory for TSP benchmark instances
├── results/           # Suggested output directory (see Usage for details)
├── requirements.txt   # Python dependencies
└── README.txt         # This file


- **`tsp_instances/`**: Place TSP files here (e.g., download from TSPLIB).
- **`results/`**: A suggested directory for outputs. For **single TSP file analysis**, outputs are saved in the same directory as the input TSP file. For **experiments**, outputs are saved to the current working directory unless `--output-dir` is specified.

## ZENODO DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15000490.svg)](https://doi.org/10.5281/zenodo.15000490)

**DOI:** [DOI: 10.5281/zenodo.15006676](https://doi.org/10.5281/zenodo.15006676)

## REQUIREMENTS

### Software
- **Python 3.8 or higher**: Core language requirement.

### Required Libraries
Install these via `requirements.txt` (pip install -r requirements.txt):
- **NumPy 1.20+**: Numerical computations.
- **Pandas 1.3+**: Data manipulation and analysis.
- **Matplotlib 3.5+**: Visualization generation.
- **NetworkX 2.6+**: Graph operations for TSP.
- **SciKit-learn 1.0+**: Clustering (KMeans) and meta-learning (RandomForestRegressor).

### Optional Libraries
Install these for additional features:
- **PyTorch (Torch)**: Hardware acceleration (CUDA for NVIDIA GPUs, Metal for Apple Silicon).
  - `pip install torch`
- **BioPython**: Genetic sequence analysis.
  - `pip install biopython`
- **xarray**: Weather data processing (requires scipy for peak detection).
  - `pip install xarray scipy`
- **psutil**: Memory usage monitoring (optional for performance tracking).
  - `pip install psutil`

### Development Environment
- Developed on a **MacBook Pro M3 Max, macOS 15.3.1**.
- Tested with Python 3.11, but 3.8+ should suffice.
- Uses standard library modules: `os`, `time`, `math`, `json`, `logging`, `argparse`, `random`, `multiprocessing`, etc.

## INSTALLATION

Follow these steps in your terminal to set up the environment:

1. **Download the Project Files**
   - Go to the arXiv page for the paper “Bridging Pattern-Aware Complexity with NP-Hard Optimization: A Unifying Framework and Empirical Study.”
   - Download the ancillary files (e.g., `pacf_v1.0.zip`) from the submission.
   - Extract the ZIP file to a local directory (e.g., `pacf_v1.0/`).

2. **Navigate to the Directory**
   ```bash
   cd pacf_v1.0
   ```

3. **Install Python (if not installed)**
   - Ensure Python 3.8 or higher is installed. On macOS, use `brew install python@3.11` or download from [python.org](https://www.python.org). Verify with `python3 --version`.

4. **Create a Virtual Environment** (Recommended)
   ```bash
   python3 -m venv pacf_env
   ```

5. **Activate the Virtual Environment**
   - macOS/Linux: `source pacf_env/bin/activate`
   - Windows: `pacf_env\Scripts\activate`

6. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

7. **Install Optional Dependencies** (as needed)
   - For hardware acceleration: `pip install torch`
   - For genetic sequences: `pip install biopython`
   - For weather data: `pip install xarray scipy`
   - For memory monitoring: `pip install psutil`

8. **Verify Setup**
   - Run: `python PACF_v1.py --version`
   - Expected output: `PACF v1.0`

...

## Code Availability
The source code is available on:
- HAL: [https://hal.science/oliviersaidi](https://hal.science/oliviersaidi)
- Zenodo: [https://doi.org/10.5281/zenodo.15000490](https://doi.org/10.5281/zenodo.15000490)
- GitHub: [https://github.com/oliviersaidi/pacf-framework](https://github.com/oliviersaidi/pacf-framework)

An arXiv submission is planned once the author’s account is validated.

## Citation
```bibtex
@misc{saidi2025bridging,
   title={Bridging Pattern-Aware Complexity with NP-Hard Optimization: A Unifying Framework and Empirical Study},
   author={Saidi, Olivier},
   year={2025}
}
```
