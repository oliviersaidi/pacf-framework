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

## USAGE

The script supports three main modes via the command line:

### 1. Single TSP File Analysis
Analyze a specific TSP instance and generate a four-panel visualization.

**Command:**
```bash
python PACF_v1.py tsp_instances/<instance_name>.tsp
```

**Example:**
```bash
python PACF_v1.py tsp_instances/att48.tsp
```

**Output:**
- Console logs and metrics (tour length, runtime, PUE, SQF).
- Visualization saved as `<input_directory>/<instance_name>_analysis.png` (e.g., `tsp_instances/att48_analysis.png`).
- Results saved as `<input_directory>/<instance_name>_results.json` (e.g., `tsp_instances/att48_results.json`).

**Note**: Outputs are saved in the same directory as the input TSP file (e.g., if the input is `tsp_instances/att48.tsp`, outputs go to `tsp_instances/`).

### 2. Get Dataset Suggestions
List recommended benchmark datasets.

**Command:**
```bash
python PACF_v1.py suggest-datasets
```

**Output:**
- Console output with dataset names, URLs, and descriptions for TSP, genetic, and weather domains.

### 3. Run Experiment
Run multiple solvers on a dataset.

**Command:**
```bash
python PACF_v1.py run-experiment --problem tsp --dataset tsp_instances/
```

**Options:**
- `--problem`: `[tsp, genetic, weather]` (required).
- `--dataset`: Path to dataset file or directory (required).
- `--output`: Results file (default: `results.json`).
- `--format`: `[auto, tsplib, fasta, netcdf, csv]` (default: `auto`).
- `--parallel`: Enable parallel processing.
- `--sequential`: Disable parallel processing.
- `--visualize`: Generate visualizations for each instance.
- `--time-limit`: Max seconds per solver-instance pair (e.g., `30.0`).
- `--output-dir`: Directory for output files (default: current working directory).
- `--advanced-init`: Use advanced tour initialization.
- `--aggressive-search`: Use aggressive swap selection.
- `--use-4opt`: Enable 4-Opt for small instances (<200 cities).

**Output:**
- Results saved to `<output-dir>/<output>` (e.g., `results.json` in the current working directory if `--output-dir` is not specified, or in the specified `--output-dir`).
- Visualizations (if enabled with `--visualize`) saved to `<output-dir>/<instance_name>_analysis.png`.

**Note**: By default, outputs are saved to the current working directory (where you run the command from). Use `--output-dir` to specify a custom output directory (e.g., `--output-dir results` to save all outputs in `results/`).

## KEY COMPONENTS

1. **Pattern Detection**: Identifies structural patterns (e.g., clustering, symmetry).
2. **Complexity Calculation**: Adjusts complexity based on patterns:  
   `C(P,A) = T_base(n) * f(n,ρ,H) * R(P,A) + C_residual`.
3. **PUE**: Measures pattern exploitation efficiency.
4. **SQF**: Quantifies solution quality improvement over baseline.
5. **Solver Selection**: Uses meta-learning to pick optimal solvers.

## SOLVER DESCRIPTIONS

### TSP Solvers
- **Nearest Neighbor**: Greedy, O(n²), baseline for SQF.
- **2-Opt**: Local search with 2-edge swaps.
- **3-Opt**: Enhanced local search (used for <500 cities).

### Optional Solvers (Experiment Mode with Flags)
- **Enhanced 3-Opt**: 3-Opt with aggressive moves and annealing (`--aggressive-search`).
- **4-Opt**: 4-edge swaps for small instances (`--use-4opt`, <200 cities).
- **Adaptive**: Combines multiple methods (`--advanced-init` + `--aggressive-search`).

### Genetic Sequence Solver (Requires BioPython)
- **RepeatAwareAligner**: Exploits repetitive patterns.

### Weather Forecasting Solver (Requires xarray)
- **SeasonalForecaster**: Leverages seasonal patterns.

## ADVANCED FEATURES

1. **Advanced Tour Initialization**: Multi-Fragment, Convex Hull, Christofides (`--advanced-init`).
2. **Aggressive Swap Selection**: Prioritizes promising swaps (`--aggressive-search`).
3. **Meta-Learning**: Predicts best solver using RandomForestRegressor.
4. **Hardware Acceleration**: Uses PyTorch for CUDA (NVIDIA) or Metal (Apple Silicon) if available.

## PERFORMANCE METRICS

1. **Pattern Utilization Efficiency (PUE)**: % of elements covered by patterns (0-100%).
2. **Solution Quality Factor (SQF)**: % improvement over Nearest Neighbor (TSP baseline).
3. **Efficiency Index (EI)**: Runtime ratio (baseline/solver).
4. **Pattern-Aware Complexity**: Adjusted complexity reflecting structure.

## EXAMPLE WORKFLOW

1. **Load a TSP Instance**
   ```bash
   python PACF_v1.py tsp_instances/att48.tsp
   ```

2. **Run an Experiment**
   ```bash
   python PACF_v1.py run-experiment --problem tsp --dataset tsp_instances/ --visualize --output-dir results
   ```

3. **Interpret Results**
   - Check `<input_directory>/att48_results.json` for metrics (single analysis).
   - Check `results/results.json` for experiment metrics.
   - View visualizations in `<input_directory>/` or `results/`.

## TROUBLESHOOTING

1. **Memory Errors**
   - Increase memory or use `--sequential` for large instances.
   - Reduce `parallel_batch_size` in `SolverConfiguration` (edit `PACF_v1.py`).

2. **Slow Performance**
   - Enable `--parallel` or reduce `--time-limit`.
   - Use `--advanced-init` and `--aggressive-search`.

3. **Missing Dependencies**
   - Verify `requirements.txt` installation.
   - Install optional libraries for genetic/weather features.

4. **Logging**
   - Add `--log-level DEBUG` for detailed logs:
     ```bash
     python PACF_v1.py tsp_instances/att48.tsp --log-level DEBUG
     ```
   - Use `--log-level WARNING` to reduce output.

5. **File Not Found**
   - Ensure TSP files are in `tsp_instances/`.
   - Download from [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

## CITATION

If used in research, please cite:

@misc{saidi2025bridging,
  title={Bridging Pattern-Aware Complexity with NP-Hard Optimization: A Unifying Framework and Empirical Study},
  author={Saidi, Olivier},
  year={2025}
}

## Author
- **Olivier Saidi**
   - Location: Paris, France
   - HAL: [https://hal.science/oliviersaidi](https://hal.science/oliviersaidi)
   - ORCID: [0009-0004-3221-6911](https://orcid.org/0009-0004-3221-6911)
   - ResearchGate: [Olivier-Saidi-2](https://www.researchgate.net/profile/Olivier-Saidi-2)
   - GitHub: [oliviersaidi/pacf-framework](https://github.com/oliviersaidi/pacf-framework)
   - Zenodo: [oliviersaidi](https://zenodo.org/users/oliviersaidi)
   
## Code Availability
The source code is available on:
- HAL: [https://hal.science/oliviersaidi](https://hal.science/oliviersaidi)
- Zenodo: [https://zenodo.org/users/oliviersaidi](https://zenodo.org/users/oliviersaidi) (DOI forthcoming)
- GitHub: [https://github.com/oliviersaidi/pacf-framework](https://github.com/oliviersaidi/pacf-framework)

An arXiv submission is planned once the author’s account is validated.

## Citation
```bibtex
@misc{saidi2025bridging,
   title={Bridging Pattern-Aware Complexity with NP-Hard Optimization: A Unifying Framework and Empirical Study},
   author={Saidi, Olivier},
   year={2025}
}
