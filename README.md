# PATTERN-AWARE COMPLEXITY FRAMEWORK (PACF) v1.0

[![CI](https://github.com/oliviersaidi/pacf-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/oliviersaidi/pacf-framework/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18924030.svg)](https://zenodo.org/records/18924030)
[![arXiv](https://img.shields.io/badge/arXiv-2506.13810-b31b1b.svg)](https://arxiv.org/abs/2506.13810)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)

> 📄 **Paper**: [Bridging Pattern-Aware Complexity with NP-Hard Optimization: A Unifying Framework and Empirical Study](https://arxiv.org/abs/2506.13810) — Olivier Saidi, 2025 · arXiv:2506.13810 [cs.AI]

The TSP data used in this work is from [TSPLIB - University of Heidelberg](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), which is freely available for research purposes.

## OVERVIEW

The **Pattern-Aware Complexity Framework (PACF)** is a Python-based framework designed to analyze and exploit structural patterns in NP-hard optimization problems, such as the Traveling Salesman Problem (TSP), genetic sequence alignment, and weather forecasting. By identifying patterns like symmetry, clustering, and repetition, PACF reduces computational complexity and enhances solver performance. This implementation focuses on TSP but is extensible to other domains with optional dependencies.

Developed and tested on a **MacBook Pro M3 Max running macOS 26.3.1**, the framework should work across platforms (macOS, Linux, Windows) with the appropriate setup.

## DIRECTORY STRUCTURE

```
pacf-framework/
├── PACF_v1.py            # Main script
├── requirements.txt      # Python dependencies (pip install -r requirements.txt)
├── test_pacf.py          # Smoke tests
├── TSP_instances/        # 24 TSPLIB benchmark instances (ready to use)
├── .github/
│   ├── workflows/
│   │   └── ci.yml        # CI: Python 3.9 + 3.11 on every push/PR
│   └── ISSUE_TEMPLATE/
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── SECURITY.md
```

- **`PACF_v1.py`**: Single-file framework — run directly, no installation beyond `requirements.txt`.
- **`TSP_instances/`**: 24 benchmark instances from TSPLIB, ready to use out of the box.
- **Outputs**: For single-file analysis, saved alongside the input `.tsp` file. For experiments, saved to the current working directory unless `--output-dir` is specified.

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


## INSTALLATION

Follow these steps in your terminal to set up the environment:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/oliviersaidi/pacf-framework.git
   cd pacf-framework
   ```

2. **Install Python (if not installed)**
   - Ensure Python 3.8 or higher is installed. On macOS, use `brew install python@3.11` or download from [python.org](https://www.python.org). Verify with `python3 --version`.

3. **Create a Virtual Environment** (Recommended)
   ```bash
   python3 -m venv pacf_env
   ```

4. **Activate the Virtual Environment**
   - macOS/Linux: `source pacf_env/bin/activate`
   - Windows: `pacf_env\Scripts\activate`

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Install Optional Dependencies** (as needed)
   - For hardware acceleration: `pip install torch`
   - For genetic sequences: `pip install biopython`
   - For weather data: `pip install xarray scipy`
   - For memory monitoring: `pip install psutil`

7. **Verify Setup**
   ```bash
   python PACF_v1.py --version          # → PACF v1.0
   python PACF_v1.py TSP_instances/berlin52.tsp
   ```


## Research Publication
- arXiv: [https://arxiv.org/abs/2506.13810](https://arxiv.org/abs/2506.13810) (cs.AI)
- Zenodo: [https://doi.org/10.5281/zenodo.15006676](https://zenodo.org/records/15006676)
- GitHub: [https://github.com/oliviersaidi/pacf-framework](https://github.com/oliviersaidi/pacf-framework)



## Citation
```bibtex
@misc{saidi2025bridging,
   title        = {Bridging Pattern-Aware Complexity with NP-Hard Optimization: A Unifying Framework and Empirical Study},
   author       = {Saidi, Olivier},
   year         = {2025},
   eprint       = {2506.13810},
   archivePrefix= {arXiv},
   primaryClass = {cs.AI},
   doi          = {10.48550/arXiv.2506.13810},
   url          = {https://arxiv.org/abs/2506.13810}
}
```
