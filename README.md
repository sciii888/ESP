# ESP
# 1.Raster sampling script

This script samples balanced pixels from raster features using two binary masks
(positive and negative) and saves the results to a CSV file.

- Input features: ./data/features/*.tif  
- Masks: ./data/masks/source00.tif, ./data/masks/low00.tif  
- Output: ./data/samples/sample1.csv  

Run:

```bash
python sampling_script.py

# 2.train—_model

This script trains a binary XGBoost model on the sampled data and saves the
trained model and evaluation metrics.

- Input: `./data/samples/sample1.csv`  
- Outputs:  
  - `./models/run_0/xgb_model.pkl`  
  - `./models/run_0/scaler.pkl`  
  - `./models/run_0/model_metrics.csv`  
  - `./models/run_0/cross_validation.csv`  

Run:

```bash
python train_model.py

#3. Resistance surface prediction script

This script loads a trained XGBoost model and feature rasters, predicts
per-pixel resistance (as class-1 probability), and saves the result as a GeoTIFF.

- Input features: `./data/features/*.tif`
- Model file: `./models/run_0/xgb_model.pkl`
- Output: `./outputs/resistance.tif`

Run:

```bash
python predict_resistance.py
