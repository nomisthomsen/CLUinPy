# Suitability Modelling Manual (prerequisite for CLUinPy)

*Authors:* Simon Thomsen & Melvin Lippe  
*Contact:* simon.thomsen@thuenen.de  
*Module:* `suitability`  
*Runner script:* `src/scripts/run_suitability.py`  

This manual explains how to create **land-cover suitability maps** with the `suitability` module. The output is a **multi-band raster stack** (one suitability layer per land-cover class) that can be used as an input to the land-use change model in **CLUinPy**.

The workflow implemented in `suitability.main.suitability()` includes:

- optional feature selection using **correlation sampling** and **variance inflation factor (VIF)** filtering
- **stratified sampling** from the land-cover raster (or reading samples from a point shapefile)
- model training using one or more algorithms
- model evaluation (AUC) and model logs
- prediction and export of suitability rasters (optionally for **dynamic predictor years**)
- optional **ensemble** prediction

---

## 1. Key concepts

### 1.1 What is produced?
Depending on your settings, the module produces:

- **Suitability stacks** (GeoTIFF): one band per land-cover class (class IDs from the classification raster)
- **Model logs** (text/metadata): AUC scores and model-specific diagnostics
- Optional: **Ensemble** suitability stack (if enabled)
- Optional: Suitability predictions for **dynamic years** (if enabled)

### 1.2 How are land-cover classes handled?
The land-cover raster passed as `classification` defines:

- which land-cover classes exist
- the **order** of classes used for training/prediction outputs

**Important rule:** the code treats all values **>= 0** as valid classes and ignores negative values (and also ignores your `no_data_value`).  
So your land-cover classes should be encoded as **0..N** (or at least non-negative integers), and no-data should be a distinct value (commonly `-9999`).

---

## 2. Input data requirements

This section is the most important part of running suitability successfully.

### 2.1 Required inputs

#### A) Land-cover raster (`classification`)
A single-band raster that provides the **observed land-cover / land-use class** per pixel.

**Requirements**
- File path readable by `rasterio` (typically `.tif`)
- Single band
- Integer class IDs
- Class values should be **non-negative** (>= 0)
- No-data pixels should use `no_data_value` (default `-9999`)

**Strong recommendation**
- Use consecutive classes starting at **0** (0,1,2,...), as this is also required in the land use modelling later.  

#### B) Environmental predictors (`env_vars`)
A list of raster paths (each typically a single-band GeoTIFF) used as explanatory variables.

**Requirements**
- All predictor rasters must align with `classification`:
  - same pixel size / resolution
  - same extent (or at least same array shape)
  - same CRS
  - same grid alignment (transform)
- No-data handling:
  - pixels with `no_data_value` are treated as missing. The `no_data_value` should be the same as for the land-cover raster.

**Typical predictors (exemplary):**
- Biophysical:
  - Topography (altitude, slope, aspect)
  - Climate (e.g. bioclimatic variables, also available for future climate scenarios)
  - Soil characteristics
- Socioeconomic:
  - Population density
  - Accessibility
  - Distance to markets

---

### 2.2 Optional inputs

#### C) Point shapefile with samples (`sample_points_shapefile`)
Instead of drawing a stratified random sample from the land-cover raster, you can provide a point shapefile. The module will extract predictor and land-cover values at each point.

**Requirements (enforced by code)**
- Shapefile geometries must be valid
- Shapefile must contain **only Point geometries**
- If CRS differs from the raster CRS, it will be reprojected automatically
- Points outside the raster extent are silently skipped

**What attributes are required?**
None. The script **does not** read attributes for the class label; it extracts the land-cover value from the raster at the point location.

> In case the user prefers to use a point shapefile with samples, please be advised to think about a suitable sampling strategy, which considers spatial auto-correlation and different class sizes.

#### D) Dynamic predictor variables (`dynamic`, `dyn_years`, `dyn_vars`)
Dynamic mode allows you to **swap predictors** for certain future years and generate suitability maps for those years. This can be particularly relevant for predictor variables which are also available as future projections (e.g. climate or population).

**How it works conceptually**
- You train on the baseline predictors in `env_vars`
- For prediction years in `dyn_years`, the code replaces some predictors with their year-specific versions from `dyn_vars`

**Requirements**
- `dynamic=True`
- `dyn_years`: list of years, e.g. `[2030, 2040, 2050]`
- `dyn_vars`: list of *dynamic predictor rasters* corresponding to variables in `env_vars`  
  In your original manual, dynamic rasters follow the naming convention:
  - baseline: `Bioclim01.tif`
  - dynamic: `Bioclim01_2030.tif`  
  and years must match entries in `dyn_years`.


---

## 3. Parameters of `suitability()`

The runner script `run_suitability.py` calls `suitability.main.suitability()` with the parameters below.

### 3.1 Parameter overview (with typical values)

| Parameter | Type | Required | Example / default                                                        | Meaning |
|---|---:|:--------:|--------------------------------------------------------------------------|---|
| `classification` | `str` |   Yes    | path to `landcover.tif`                                                  | Land-cover raster (training labels) |
| `env_vars` | `List[str]` |   Yes    | list of `.tif` paths                                                     | Predictor rasters |
| `mode` | `str` or `List[str]` |   Yes    | One or multiple of `['random_forest','XGBoost','logistic', 'SVM', 'MLP']` | Model(s) to fit |
| `out_path` | `str` |   Yes    | output folder                                                            | Where to write results |
| `n_samples_corr` | `int` |   Yes    | `1000`                                                                   | Samples for correlation/VIF step |
| `vif_threshold` | `int` |    No    | `5`                                                                      | Filter collinear predictors |
| `min_distance` | `int` |    No     | `1`                                                                      | Minimum pixel distance between sampled points |
| `test_fraction` | `float` |    No     | `0.3`                                                                    | Hold-out share for testing |
| `random_state` | `int` or `None` |    No     | `None`                                                                   | Reproducible randomness |
| `sample_points_shapefile` | `str` or `None` |    No     | `None`                                                                   | Provide sampling points (skip raster sampling) |
| `sample_size_list` | `List[int]` or `None` |    ⚠️    | computed                                                                 | Samples per class if no shapefile |
| `dynamic` | `bool` |    No     | `False`                                                                  | Enable dynamic prediction years |
| `dyn_years` | `List[int]` or `None` |    No     | `None`                                                                   | Years for dynamic predictions |
| `dyn_vars` | `List[str]` or `None` |    No     | `None`                                                                   | Dynamic predictor rasters |
| `ensemble` | `bool` |    No     | `False`                                                                  | Compute ensemble prediction |
| `no_data_value` | `int`/`float` |    No     | `-9999`                                                                  | No-data value used in rasters |
| `predict_outputs` | `bool` |    No     | `True`                                                                   | If `False`, only train + evaluate |

⚠️ **Important dependency:**  
If you **do not** provide `sample_points_shapefile`, then you **must** provide `sample_size_list`. The function raises an error otherwise.

---

### 3.2 Detailed parameter explanations

#### 3.2.1 `mode`
You can pass:
- a single model name: `"random_forest"`
- or a list: `["random_forest", "XGBoost", "logistic"]`

Valid names are:
- `logistic` (Logistic Regression)
- `random_forest` (Random Forest regressor)
- `XGBoost` (Extreme Gradient Boosting)
- `SVM` (Support Vector Machine)
- `MLP` (Multilayer Perceptron)

You can find 

#### 3.2.2 `n_samples_corr` and `vif_threshold`
The module draws `n_samples_corr` samples to estimate collinearity. Predictors with VIF above `vif_threshold` are removed.

Practical guidance:
- If you have many predictors (>30), keep `n_samples_corr` relatively large (e.g., 1000–5000)
- If you have few predictors (<10), 500–1000 is often enough
- Common threshold for VIF is 10, while a threshold of 5 is also common but more restrictive

#### 3.2.3 `sample_size_list`
A list of per-class sample counts for training/validation. The list must correspond to the **sorted unique classes** found in the land-cover raster (all values >= 0).

In the runner script, it is computed with:

```python
sample_list = sample_per_class(lc_array, -9999, 'fraction', 0.1, 100, 500)
```

Meaning:
- sample 10% of each class
- but at least 100 and at most 500 per class

Following this approach, it is assured that classes are represented in accordance to their prevalence in the land cover map, but no classes are particularly under or overrepresented. 

#### 3.2.4 `min_distance`
Minimum distance (in pixels) between sampled points per class. Larger values enforce more spatial independence but may reduce sample availability for small classes.

#### 3.2.5 `predict_outputs`
- `True`: train, evaluate, and write suitability rasters
- `False`: train and evaluate only (useful for testing model choices quickly)

---

## 4. How to run suitability from `run_suitability.py`

The runner script is a runnable example. It:

1. loads the land-cover raster
2. computes a sampling plan (`sample_per_class`)
3. finds predictors in a folder (`find_files`)
4. calls `suitability(...)`

### 4.1 Example (as in the repository)

```python
import rasterio
from suitability.main import suitability
from suitability.sampling import sample_per_class
from suitability.io_utils import find_files

lc_path = '../testdata/rasterdata/lc2016.tif'
lc_array = rasterio.open(lc_path).read(1)

sample_list = sample_per_class(lc_array, -9999, 'fraction', 0.1, 100, 500)

pred_vars = find_files('testdata/rasterdata/pred_variables', '.tif', '.tif')

suitability(
  classification=lc_path,
  env_vars=pred_vars,
  mode=['random_forest', 'XGBoost', 'logistic'],
  out_path='../testdata/rasterdata/pred_variables/suitability_out',
  n_samples_corr=1000,
  vif_threshold=5,
  min_distance=3,
  test_fraction=0.3,
  random_state=12,
  sample_size_list=sample_list,
  no_data_value=-9999,
  predict_outputs=True
)
```

### 4.2 Common pitfalls checklist

Before running, check:

- **Alignment**: all predictor rasters align with `classification` (same shape/transform/CRS)
- **Class IDs**: land-cover classes are non-negative; no-data is `no_data_value`
- **Sampling**:
  - either provide `sample_points_shapefile`
  - or compute/provide `sample_size_list`
- **Output folder** exists or can be created (permissions)
- **Mode names** exactly match allowed strings

---

## 5. Outputs

The code writes outputs to a timestamped directory under `out_path`. Typical outputs include:

- Suitability stack(s) per model (`.tif`)  
  Bands correspond to land-cover classes.
- Model evaluation/log files (`.txt` or metadata)
- Optional ensemble prediction outputs

---

## 6. Quick guidance on choosing parameters

- Start with `mode=['random_forest']` for a robust and quick baseline.
- Use `logistic` if interpretability (coefficients) is important.
- Set `min_distance` > 1 if spatial autocorrelation is strong and you want more independence. This may also depend on your raster resolution.
- Increase `n_samples_corr` if VIF filtering behaves unstable between runs.

