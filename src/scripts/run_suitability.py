import rasterio

from suitability.main import suitability
from suitability.sampling import sample_per_class
from suitability.io_utils import find_files

# This script serves to run suitability calculation, which can be used in src afterward.
# You may need to adapt the paths to your local folder structure.


# Load land cover raster as array
lc_path = 'testdata/rasterdata/lc2016.tif'
lc_array = rasterio.open('testdata/rasterdata/lc2016.tif').read(1)

# Calculate number of points to be sampled per class (stratified random sample)
sample_list = sample_per_class(lc_array, -9999,'fraction', 0.1, 100, 500)

# Load predictor variables
pred_vars = find_files('testdata/rasterdata/pred_variables','.tif','.tif')

suitability(classification=lc_path,
            env_vars=pred_vars,
            mode = ['random_forest','XGBoost','logistic'],
            out_path='testdata/rasterdata/pred_variables/suitability_out',
            n_samples_corr=1000,
            vif_threshold=5,
            min_distance=3,
            test_fraction=0.3,
            random_state=12,
            sample_size_list=sample_list,
            no_data_value=-9999,
            predict_outputs=True)