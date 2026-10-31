from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "production_pipeline"

model = keras.models.load_model(ARTIFACTS_DIR / "churn_ann.keras")
preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")

metadata = json.loads(
    (ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8")
)

RAW_INPUT_COLUMNS = metadata["raw_input_columns"]
THRESHOLD = float(metadata["threshold"])


def predict_customer(customer: dict) -> dict:
    """Predict churn for one customer using raw customer fields."""
    customer_df = pd.DataFrame([customer])

    missing = sorted(set(RAW_INPUT_COLUMNS) - set(customer_df.columns))
    extra = sorted(set(customer_df.columns) - set(RAW_INPUT_COLUMNS))

    if missing or extra:
        raise ValueError(
            f"Missing columns: {missing}; unexpected columns: {extra}"
        )

    customer_df = customer_df[RAW_INPUT_COLUMNS]
    processed = preprocessor.transform(customer_df).astype(np.float32)

    probability = float(model.predict(processed, verbose=0).ravel()[0])
    predicted_class = int(probability >= THRESHOLD)

    return {
        "churn_probability": round(probability, 3),
        "prediction": "Churn" if predicted_class else "No churn",
        "threshold": round(THRESHOLD, 3),
    }