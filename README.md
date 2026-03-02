# CLUinPy: CLUMondo-Based Land Use Modeling in Python

This Python-based land use modeling script simulates land use and land cover changes (LULCC) using principles inspired by the original **CLUMondo C++ model** created by **Peter Verburg** (link to the [Github repository](https://github.com/VUEG/CLUMondo) and link to the publication by [van Asselen & Verburg 2012](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12331)). 

This repository contains three sections:

1. The location suitability modelling (which is a prerequisite for land use modelling in CLUMondo / CLUinPy). You can find the relevant scripts in the folder [suitability](src/suitability/) and the manual [here](docs/suitability_manual.md).
  
2. The proper LULCC model CLUinPy based on CLUMondo. You can find the relevant scripts in the folder [cluinpy](src/cluinpy) and the manual [here](docs/CLUinPy_manual.md).

3. A beginners [tutorial](docs/Tutorial_test_data.md), including test data for Loreto canton, Ecuador.

---

## Repository Structure (important)

The Python source code lives inside the `src/` folder.

- **To run the model without installing the package**, change directory into `src/` first (see Quickstart below).  
- Alternatively, install the repo in editable mode (`pip install -e .`) and run from anywhere.


## Requirements

- Python **3.8+** (recommended: **3.10/3.11**)
- Recommended environment manager: **Miniforge / Anaconda / Mambaforge** (Conda-based)

### Required Python packages

Core dependencies include:
- `numpy`, `pandas`, `geopandas`
- `rasterio`, `gdal`
- `openpyxl`
- `numba`, `joblib`
- `scikit-learn`, `statsmodels`, `scipy`, `xgboost`

> Note: Geospatial packages (`gdal`, `rasterio`, `geopandas`) are often easiest to install via **conda-forge**.

---
## Installation

### Recommended option: Conda / Miniforge (geospatial-friendly)

Create an environment (example name: `clupy`) and install dependencies.

```bash
conda create -n clupy python=3.10
conda activate clupy
```

Install dependencies (recommended via conda-forge for geospatial packages):
```bash
conda install -c conda-forge numpy pandas geopandas rasterio gdal openpyxl numba joblib scikit-learn
```

Optional (recommended for development / stable imports):
```bash
pip install -e .
```
---
## Quickstart: Run CLUinPy
The main entrypoint is:
- `src/scripts/run_CLUinPy.py`
The script calls model logic from:
- `src/cluinpy/`

### 1) Run without installing (simple)
**Always run from the `src/` directory** (the folder containing `cluinpy/`, `scripts/`, `suitability/`)

**Windows (Powershell/ Windows Terminal):**
```powershell
cd "\path\to\repo-root\src"
conda run --no-capture-output -n clupy python -m scripts.run_CLUinPy --config "path\to\config_file.txt"
```

**Windows (Anaconda Prompt/ Miniforge Prompt):**
```bat
cd \path\to\repo-root\src
conda activate clupy
python -m scripts.run_CLUinPy --config "path\to\config_file.txt"
```

**Linux/ HPC (bash):**
```bash
cd /path/to/repo-root/src
conda activate clupy
python -m scripts.run_CLUMondoPy --config "/path/to/config_file.txt"
```

### 2) Run after installing (`pip install -e .`)
If you installed in editable mode, you can still run using -m (recommended), and you are less sensitive to the current directory:
```bash
python -m scripts.run_CLUinPy --config "/path/to/config_file.txt"
```
---
## Configuration

The model is executed using a --config text file. The config typically contains:

- file paths to required inputs (rasters, Excel tables, etc.)
- model/scenario parameters
- output paths or output directory settings

Please refer to the [manual](docs/CLUinPy_manual.md) for a complete list of input requirements for the configuration txt file.

### Path notes

**Windows:** quote paths, especially if they contain spaces.

Both `E:\folder\file.txt` and `E:/folder/file.txt` are typically fine for Python.

---

## Suitability Modelling
CLUinPy requires users to include location suitability maps for each land cover class. This repository offers users to model location suitability with a dedicated module. The code can be found in `src/suitability/` and the core script to run the model is `run_suitability.py`.   

In our modelling framework, users can choose between different modelling algorithms, including Logistic Regression, Random Forest and Support Vector Machines. More information on suitability modelling can be found in the respective [manual](docs/suitability_manual.md). 

---

## Authors
- Simon Thomsen (simon.thomsen@thuenen.de)  
- Melvin Lippe (melvin.lippe@thuenen.de)

## License
This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html). Please refer also to the [LICENSE](src/LICENSE.md) file in this repository.


