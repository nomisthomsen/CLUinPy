# CLUinPy

*Authors:* Simon Thomsen & Melvin Lippe  
*Contact:* simon.thomsen@thuenen.de  
*Module:* `Land_use_model`  
*Runner script:* `CLUinPy/scripts/run_CLUinPy.py`  

> This document is the manual for the CLUinPy model. It explains the model functioning but most importantly the necessary data and parameters to successfully run the model. We have provided a [hands-on tutorial](Tutorial_test_data) and [test data](../testdata) apart in another document / folder. 

> Important: The content and code of the CLUinPy model is largely **based on CLUMondo**, which was released in C++ by Peter Verburg (https://github.com/VUEG/CLUMondo). We decided to give this Python adaptation of CLUMondo an own name to avoid confusion regarding authorship, endorsement, or institutional affiliation. We acknowledge the legacy of CLUMondo explicitly in the copyright statement. In previous versions, this model had the working title *CLUMondoPy*.   


---

## Overview

CLUinPy simulates land-use / land-cover (LULC) change based on the core ideas of the original CLUMondo model:

- **Demand-driven allocation:** land-use change is driven by externally defined demand for one or more *services* (e.g., food, timber, housing).
- **Iterative allocation:** within each simulated year, the model iterates until supply–demand differences fall below tolerances or `max_iter` is reached.
- **Transition potential:** the probability ("potential") for a cell to transition into a land-use class or remain in its current land-use class is computed from:
  - **location suitability** (per class),
  - **neighbourhood influence** (per class),
  - **conversion rules** (allowance matrix, zonal constraints, age constraints, autonomous change),
  - **conversion resistance** (inertia for staying in the same class),
  - **demand elasticities** (updated during the iterative allocation).

This repository provides two ways to run the model:

1. **Direct command line arguments**
2. **A config file** that lists arguments in `--arg=value` format

---

## Requirements

### Python
- Python **3.8+** (as stated in existing documentation; check `requirements.txt` in the repo for exact dependencies).

### Core packages
The CLUinPy runner relies on typical geospatial + scientific packages:

- `numpy`, `pandas`
- `rasterio`
- `gdal` / `osgeo`
- `openpyxl`
- `numba`

---

## How the model allocates change

The model operates on **integer-coded land-use classes** (0..N-1) and uses a **stack** of suitability and neighbourhood layers (one layer per class). For each cell, it evaluates every possible target class and picks the class with the highest "potential".

A simplified view of the potential used in `src/cluinpy/transitions.py`:

- For a candidate target class `i`, the model combines:
  - `suitability[i]`
  - `preference[i]` *(optional)*
  - `neighbourhood[i]`
  - `demand_elasticity[d]` *(d is the corresponding service demand for class i according to `lus_conv`)*
  - plus **conversion resistance** `conv_res[i]` **only if** the cell stays in the same class (`old_cov == i`) — this acts like inertia.

Conversions can be restricted or enabled via:

- `region_array` (no change zones, e.g. protected areas)
- `allow` matrix (class-to-class rules)
- `zonal_array` (allow only in mapped zones for specific transitions)
- `age_array` (allow / forbid after a certain age; also supports autonomous change)

---

## Input data & parameters

You can provide arguments either via:

- **CLI:** `python -m CLUinPy.scripts.run_CLUinPy --land_array ...`  
  *(or run the script file directly)*
- **Config file:** `python .../run_CLUinPy.py --config config.txt`

### Important conventions

#### Land-use class indexing
- **Class IDs must start at 0** and increase sequentially (`0, 1, 2, ..., N-1`).
- **No gaps** are allowed.

If your land-use map uses other codes (e.g., CORINE codes), reclass it first.

#### Raster alignment
All rasters must match the land-use raster in:
- number of rows/cols
- georeferencing / transform
- extent
- CRS

The model assumes pixel-perfect alignment.

#### NoData
- Internally, the model treats `no_data_value` (default `-9999`) as invalid.
- Cells where `suit_array` has `no_data_value` are forced into `region_array == 1` (no change).
- Output rasters replace internal NoData with `no_data_out`.

---

## Arguments in `run_CLUinPy.py`

The runner script parses arguments and then calls:

`src/cluinpy/model.py::clu_dynamic(...)`

Below is a detailed guide to all arguments.

### Required arguments (in practice)

The script does not set them as `required=True` in argparse, but the model **will not run** without them.

| Argument | Type | Meaning | Expected shape / format |
|---|---:|---|---|
| `--land_array` | str | Path to initial land-use raster (single band) | GeoTIFF; values = class IDs (0..N-1) |
| `--suit_array` | str | Path to suitability stack raster | Multi-band GeoTIFF; **N bands**; values typically 0..1 |
| `--region_array` | str | Path to region restriction raster | Single band; `1` = restricted/no-change; `0` = eligible |
| `--neigh_weights` | str | Comma-separated weights, one per class | length N; e.g. `0.2,0.3,0.5` |
| `--start_year` | int | First simulation year (inclusive) | e.g. 2020 |
| `--end_year` | int | Last simulation year (inclusive) | e.g. 2030 |
| `--demand` | str | Path to demand table (Excel) | rows = years; cols = services |
| `--dem_weights` | str | Service priority weights | length = number of services |
| `--lus_matrix_path` | str | Service yield matrix (Excel) **or** folder of yearly matrices | see details below |
| `--lus_conv` | str | Conversion priority / multifunctionality matrix | Excel; rows = classes; cols = services |
| `--allow` | str | Allowance matrix for class-to-class transitions | Excel; rows/cols = classes |
| `--conv_res` | str | Conversion resistance (inertia) per class | length N |
| `--out_dir` | str | Output directory | folder path |
| `--crs` | str | CRS string used for output | e.g. `EPSG:4326` |

> **Where more information may be needed:** the runner assumes a strict mapping between the **order of classes** and the **band order** in the suitability stack, and the **row/column order** in the Excel matrices. If you document class order in a dedicated table (labels ↔ IDs), users will make fewer mistakes.

---

## Detailed explanation of each input

### 1) `land_array` (initial land-use / land-cover map)

**Argument:** `--land_array`  
**Type:** single-band raster

**What it is:** The initial land-use map at the start of the simulation.

**Required format**
- integer values **0..N-1** (no gaps)
- NoData should be `no_data_value` (default `-9999`) *or* be reclassified to that value

**How it is used**
- Provides the starting state (`old_cov`) for the first year
- Drives neighbourhood influence (neighbourhood is computed as class share in a moving window)
- Used to compute current service supply per year (via class counts × `lus_matrix`)

---

### 2) `suit_array` (location suitability stack)

**Argument:** `--suit_array`  
**Type:** multi-band raster stack (one band per class)

**What it is:** Suitability scores per land-use class.

**Required format**
- **N bands**, where `band i` corresponds to class ID `i`
- same extent/shape as `land_array`
- values typically in **0..1** (recommended), but not strictly enforced

**How it is used**
- In each iteration, per cell and target class `i`, the potential includes `suit_values[i]`.

**Dynamic suitability (optional)**
If suitability changes over time, you can update it in specific years using `change_years` + `change_paths`.

> For more detailed information location suitability modelling please refer to the [manual](suitability_manual.md) in this repository.

---

### 3) `region_array` (restricted / no-change mask)

**Argument:** `--region_array`  
**Type:** single-band raster

**What it is:** Mask where cells are blocked from changing.

**Expected values**
- `1` → restricted / no change
- `0` → eligible to change

**How it is used**
- Any cell with `region_array == 1` is frozen (keeps its old class).
- Additionally, cells where `suit_array` has NoData are forced to restricted.

---

### 4) Neighbourhood influence

#### 4.1) `neigh_weights`

**Argument:** `--neigh_weights`  
**Type:** comma-separated float list

**What it is:** A per-class multiplier applied to the neighbourhood fraction.

The neighbourhood module (`CLUMondoPy/CLUMondo/neighbourhood.py`) computes, for each class `i`:

- a binary mask of class locations
- the fraction of the window occupied by class `i`
- multiplied by `neigh_weights[i]`

So a weight of 0 disables neighbourhood effects for a class.

#### 4.2) `width_neigh`

**Argument:** `--width_neigh`  
**Type:** int, default `1`

**What it is:** Defines the window size:

- `width_neigh = 1` → 3×3
- `width_neigh = 2` → 5×5
- in general: `(1 + 2*width_neigh)²`

**How it is used**  
Neighbourhood influence is added to the potential for each target class. 

---

### 5) Conversion resistance (`conv_res`)

**Argument:** `--conv_res`  
**Type:** comma-separated float list

**What it is:** A per-class inertia term.

**How it is used**
- Only applied when a cell *stays* in its current class (`old_cov == i`).
- Larger values make persistence more likely.

**Practical tip**
- Conversion resistance values should range between 0 (no resistance) to 1 (full resistance).
- If you disabled any possible transition of a particular class to another via the `allow` matrix, the conversion resistance is somewhat obsolete but still needs to be accounted for.

---

## Tabular inputs (Excel)

All Excel matrices are read with `pandas.read_excel(...)` and some then sliced as `iloc[:, 1:]`, meaning the **first column is assumed to be labels** (class names or service names). The slicing is done to only load numerical values and convert them from pandas to numpy.

### 6) External demand table 
#### 6.1) `demand`
**Argument:** `--demand`  
**Type:** Excel file

**What it is:** A time series of demanded services.

**Expected format**
- rows = years/time steps (must cover `start_year..end_year` in order)
- columns = services (e.g. `Milk`, `Wheat`, `Timber`)
- values = demanded amount per service per year (unit is user-defined)

**Important:** the runner reads `demand = pd.read_excel(...).to_numpy()`, so it assumes the Excel file contains only numeric values or that any non-numeric columns are removed. Also, columns should only represent services demands (so no column with years/ time steps).

#### 6.2) `dem_weights`

**Argument:** `--dem_weights`  
**Type:** comma-separated float list

**What it is:** Per-service weights that scale demand elasticities, controlling how strongly each service influences land allocation.

- length must match number of demand columns
- higher weight = stronger pressure to match that service

---

### 7) Land system service matrix (`lus_matrix_path`)

**Argument:** `--lus_matrix_path`  
**Type:** either:

1. **A single Excel file** (`.xlsx`) with a service yield matrix, or
2. **A folder path** containing yearly matrices named:  
   `yield_data_{year}.xlsx` (e.g., `yield_data_2025.xlsx`)

**What it is:** Defines how much of each service each land-use class produces. In the context of agricultural land classes, it can be interpreted as productivity or yield. In case you assume that service provision per class remains stable over time, provide a single Excel file to the function. If you want service provision to be dynamic and change over time, you can point to the folder path where the Excel files for each time step are stored.

**Expected format (Excel)**
- rows = land-use classes (in class ID order)
- columns = services (must match the demand table service order)
- first row/column can contain labels; numeric matrix must start at column

**Important**: The order of services must match the order of services in the `demand` matrix. The order of land classes must match the sequence (0,1,[...],N) in the initial land-use map. 

**How it is used**
- The model computes supply as:  
  `demand_cur = Σ_class (lus_matrix[class] * class_pixel_count)`

**Multifunctionality**:   
Users can incorporate multifunctionality of land use classes through the land system service matrix. The example below is inspired by the case of land use in Loreto canton in Ecuador (see [tutorial for test data](Tutorial_test_data)). In this case, the classes pasture, perennial crops and annual crops all provide only one service each. However, the class agricultural mosaic provides all three services, though in fewer quantity than the other classes.

| Class               | Cattle | Coffee | Corn |
|---------------------|-------:|-------:|-----:|
| Forest              |      0 |      0 |    0 |
| Pasture             |      4 |      0 |    0 |
| Perennial crops     |        |     10 |    0 |
| Annual crops        |      0 |      0 |   16 |
| Agricultural mosaic |    1.5 |      2 |    4 |
|  Build-up    |      0 |      0 |    0 |
| Other  |      0 |      0 |    0 |



---

### 8) Land conversion priority matrix (`lus_conv`)

**Argument:** `--lus_conv`  
**Type:** Excel file

**What it is:** A class × service matrix used to decide whether a transition would increase or decrease service provision relative to the current class. It should have the same dimensions and structure as `lus_matrix`.

**How it is used**
In `transitions.py`, for a candidate transition `old_cov → i`, the model loops over services `d`:

- if `lus_conv[old_cov, d] < lus_conv[i, d]` then **adds** elasticity pressure for service `d`
- if `lus_conv[old_cov, d] > lus_conv[i, d]` then **subtracts** elasticity pressure

This acts like a "priority direction" matrix for multifunctional land systems.

**Multifunctionality** can be incorporated similar as in `lus_matrix`, only that instead of service provision quantity, users can insert priority ranks per demand (important, not per land use class). 


---

### 9) Allowance matrix (`allow`)

**Argument:** `--allow`  
**Type:** Excel file

**What it is:** A class × class matrix specifying which transitions are possible.

**Expected meaning**
Rows are **from** (current class), columns are **to** (target class):

|                           Value | Meaning                                                                        |
|--------------------------------:|--------------------------------------------------------------------------------|
|                             `0` | forbidden conversion                                                           |
|                             `1` | allowed conversion                                                             |
|                             `2` | allowed only in zones indicated by `zonal_array` for the **target class**      |
|                      `100 + xx` | allowed only if `age_array >= xx`                                              |
|                     `-100 - xx` | forbidden if `age_array >= xx`                                                 |
| `1000 + xx`  | activates *autonomous change mode* for transition after xx years in `age_array`|

#### Autonomous change mode
If **any** value in `allow` is greater than 1000, the model enables autonomous change logic in `age.py`. This can be useful if users want to "force" land conversion after a specific amount of time, e.g. fallow to shrubland after x years.


---

## Optional spatial inputs

### 10) `age_array`

**Argument:** `--age_array`  
**Type:** single-band raster (integer)

**What it is:** Per-cell age of the current class (years since last change), used for age-conditioned transitions and autonomous change.

**How it is used**
- enables the age-based rules in `allow`
- if autonomous change mode is active, can trigger forced conversions after a threshold age

---

### 11) `zonal_array`

**Argument:** `--zonal_array`  
**Type:** multi-band raster stack (N bands)

**What it is:** A per-class mask that restricts certain transitions (where `allow == 2`) to specific mapped zones.

**Expected format**
- N bands, band `i` corresponds to class `i`
- values typically 0/1, where 1 represents target zone

**How it is used**
- If `allow[from, to] == 2`, conversion is allowed only if `zonal_array[to] > 0` at that cell.

The usage of `zonal_array` can be helpful if you want expansion of a specific land use class to happen only in designated areas. For example, the expansion of mining pits in respective concession areas (so no mining expansion elsewhere allowed).

For now, `zonal_array` is only meant to encode *target* zones, future versions may include *from→to* pair-specific zones.

---

### 12) Preference zones (`preference_array`, `preference_weights`)

**Arguments:**
- `--preference_array` (stack)
- `--preference_weights` (list)

**What it is:** A per-class mask (0/1) that adds a *bonus* to suitability when the weight is > 0.

**How it is used**
The model computes per-class preference bonus as:

- `pref_values[i] = preference_array[i] * preference_weights[i]`

and then **adds** `pref_values[i]` to the transition potential.

**Practical tips**
- Use small weights (e.g., 0..1) if suitability is 0..1.
- If you want preference to act like a hard constraint, use `zonal_array` instead.

---

## Optional runtime / output controls

| Argument | Default | Meaning |
|---|---:|---|
| `--max_diff_allow` | 3.0 | Per-service max % difference threshold for convergence |
| `--totdiff_allow` | 1.0 | Total % difference threshold for convergence |
| `--max_iter` | 3000 | Max within-year allocation iterations |
| `--demand_max` | 3.0 | Caps absolute elasticity values |
| `--demand_setback` | 0.5 | Elasticity reset value if cap exceeded |
| `--dtype` | `int16` | Output raster dtype (`int8`, `int16`, `float32` are typical choices) |
| `--no_data_value` | -9999 | Internal NoData marker used in arrays |
| `--no_data_out` | -9999 | Output NoData value written to GeoTIFF |
| `--out_year` | empty | If set, write outputs only for specific year(s), comma-separated |
| `--change_years` | empty | Years in which to swap suitability stack |
| `--change_paths` | empty | Paths to suitability stacks corresponding to `change_years` |

**Note on convergence thresholds**
- The model prints and logs:
  - `maxdiff` (maximum per-service percent difference)
  - `totdiff` (total percent difference across all services)
- Iteration stops when `maxdiff <= max_diff_allow` **and** `totdiff <= totdiff_allow`.

**Note on inclusion of dynamic location suitability stacks**   

In case users have created dynamic location suitability stacks in the `Suitability` module, they include those also in the modelling process.
- `--change_years` indicates the year(s) in which suitability stacks (`--suit_array`) should be updates
- `--change_paths` indicates the filenames to the dynamic suitability stacks.  

> Please note that the order of years in `--change_years` must match the files in `--change_paths`.

---

## Output files

Outputs are written into a **timestamped subfolder** inside `out_dir`.

- `covYYYY.tif` – land cover for year YYYY
- `ageYYYY.tif` – age raster (only if `age_array` provided)
- `logfile.txt` – model logs (iterations, differences, elasticities, and a record of key inputs)

If the model reaches `max_iter` without convergence, it writes:
- `covYYYY_error.tif`

---

## Running the model

### Option 1: Using a config file

Run:

```bash
python src/scripts/run_CLUinPy.py --config config.txt
```

Example `config.txt`:

```text
--land_array=./data/land_2020.tif
--suit_array=./data/suitability_stack.tif
--region_array=./data/region_mask.tif
--neigh_weights=0.2,0.3,0.5
--start_year=2020
--end_year=2030
--demand=./data/demand_projection.xlsx
--dem_weights=1.0,0.8,0.6
--lus_conv=./data/lus_conversion.xlsx
--lus_matrix_path=./data/yield_matrices/
--conv_res=0.5,0.2,0.3
--allow=./data/allow_matrix.xlsx
--out_dir=./output/
--crs=EPSG:4326
--max_diff_allow=3.0
--totdiff_allow=1.0
--max_iter=3000
--dtype=int16
--no_data_out=-9999
--width_neigh=1
--demand_max=3.0
--demand_setback=0.5
--no_data_value=-9999
```

> **Important:** In the runner, values with commas may be parsed as lists. Keep the format exactly as shown.

---

### Option 2: Command line arguments

```bash
python src/scripts/run_CLUinPy.py \
  --land_array ./data/land_2020.tif \
  --suit_array ./data/suitability_stack.tif \
  --region_array ./data/region_mask.tif \
  --neigh_weights 0.2,0.3,0.5 \
  --start_year 2020 \
  --end_year 2030 \
  --demand ./data/demand_projection.xlsx \
  --dem_weights 1.0,0.8,0.6 \
  --lus_conv ./data/lus_conversion.xlsx \
  --lus_matrix_path ./data/yield_matrices/ \
  --conv_res 0.5,0.2,0.3 \
  --allow ./data/allow_matrix.xlsx \
  --out_dir ./output/ \
  --crs EPSG:4326
```

---

## Troubleshooting

- **No output files created**
  - Check console output and `logfile.txt` for convergence issues.
  - Try increasing `max_iter`, or relaxing `max_diff_allow` / `totdiff_allow`.

- **Array shape mismatch / indexing errors**
  - Ensure all rasters match exactly in shape and alignment.
  - Ensure suitability stack has **exactly N layers**.
  - Ensure class IDs are **0..N-1**.

- **Everything is frozen / no change happens**
  - Check `region_array` (too many 1s)
  - Check `conv_res` (too strong inertia)
  - Check `allow` (transitions forbidden)
  - Check suitability values (e.g., all zeros)

- **Weird demand behaviour**
  - Confirm service order is consistent across:
    - `demand` columns
    - `lus_matrix` columns
    - `lus_conv` columns

