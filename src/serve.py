from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'
FEATURES = ["soil_ph", "nitrogen_ppm", "phosphorus_ppm", "potassium_ppm", "soil_moisture_pct", "sunlight_hours", "temperature_c", "rainfall_mm", "seed_density", "pest_pressure"]
def get_bundle():
    if not MODEL_PATH.exists(): raise RuntimeError('Train first: python src/train.py --limit 20000')
    return joblib.load(MODEL_PATH)
def predict(values):
    bundle = get_bundle()
    row = {k: float(values.get(k, 0.0)) for k in FEATURES}
    return float(bundle['model'].predict(pd.DataFrame([row], columns=FEATURES))[0])

if __name__ == '__main__':
    print('Train first with: python src/train.py --limit 20000')
    print('Then predict with: python src/predict.py data/sample_request.json')
