"""
InteliCrop - Data Generation Module
Generates a synthetic crop recommendation dataset for demonstration purposes.
In production, replace with real agricultural data from Kaggle or government sources.
"""

import numpy as np
import pandas as pd
import os


# Crop profiles: (N, P, K, temperature, humidity, ph, rainfall)
# Each tuple: (mean_N, std_N, mean_P, std_P, mean_K, std_K,
#               mean_temp, std_temp, mean_hum, std_hum,
#               mean_ph, std_ph, mean_rain, std_rain)
CROP_PROFILES = {
    "rice": (80, 10, 48, 8, 40, 5, 23.5, 2, 82, 5, 6.5, 0.3, 230, 30),
    "maize": (78, 10, 48, 8, 20, 3, 22.5, 2, 65, 5, 6.2, 0.4, 90, 15),
    "chickpea": (40, 8, 68, 8, 80, 5, 18.5, 2, 17, 3, 7.0, 0.3, 80, 15),
    "kidneybeans": (20, 5, 68, 8, 20, 3, 20.0, 2, 22, 3, 5.8, 0.3, 110, 15),
    "pigeonpeas": (20, 5, 68, 8, 20, 3, 27.5, 2, 50, 5, 6.0, 0.4, 140, 20),
    "mothbeans": (20, 5, 48, 8, 20, 3, 28.0, 2, 48, 5, 6.8, 0.4, 50, 10),
    "mungbean": (20, 5, 48, 8, 20, 3, 28.5, 2, 85, 5, 6.7, 0.3, 50, 10),
    "blackgram": (40, 8, 68, 8, 20, 3, 30.0, 2, 65, 5, 7.0, 0.3, 70, 10),
    "lentil": (20, 5, 68, 8, 20, 3, 24.5, 2, 60, 5, 6.8, 0.4, 50, 10),
    "pomegranate": (20, 5, 10, 3, 40, 5, 21.5, 2, 90, 3, 6.5, 0.3, 110, 15),
    "banana": (100, 10, 82, 8, 50, 5, 27.0, 2, 80, 5, 6.0, 0.3, 105, 15),
    "mango": (20, 5, 28, 5, 30, 5, 31.5, 2, 50, 5, 5.8, 0.4, 95, 15),
    "grapes": (20, 5, 130, 10, 200, 15, 23.5, 3, 82, 3, 6.0, 0.4, 70, 10),
    "watermelon": (100, 10, 10, 3, 50, 5, 25.5, 2, 85, 3, 6.5, 0.3, 50, 10),
    "muskmelon": (100, 10, 10, 3, 50, 5, 28.5, 2, 92, 3, 6.3, 0.3, 25, 5),
    "apple": (20, 5, 130, 10, 200, 15, 22.5, 2, 92, 3, 6.0, 0.3, 110, 15),
    "orange": (20, 5, 10, 3, 10, 3, 22.5, 3, 92, 3, 7.0, 0.3, 110, 15),
    "papaya": (50, 8, 60, 8, 50, 5, 33.5, 2, 92, 3, 6.8, 0.3, 145, 15),
    "coconut": (20, 5, 10, 3, 30, 5, 27.0, 2, 95, 2, 6.0, 0.3, 175, 20),
    "cotton": (120, 10, 48, 8, 20, 3, 24.0, 2, 80, 5, 7.0, 0.3, 80, 15),
    "jute": (80, 10, 48, 8, 40, 5, 25.0, 2, 80, 5, 6.8, 0.3, 175, 20),
    "coffee": (100, 10, 28, 5, 30, 5, 25.5, 2, 58, 5, 6.8, 0.3, 160, 20),
}

SAMPLES_PER_CROP = 100


def generate_dataset(output_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic crop recommendation dataset.
    
    Parameters
    ----------
    output_path : str, optional
        Path to save the CSV file. If None, saves to data/Crop_recommendation.csv
    
    Returns
    -------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(42)
    rows = []

    for crop, profile in CROP_PROFILES.items():
        (mean_N, std_N, mean_P, std_P, mean_K, std_K,
         mean_temp, std_temp, mean_hum, std_hum,
         mean_ph, std_ph, mean_rain, std_rain) = profile

        for _ in range(SAMPLES_PER_CROP):
            row = {
                "N": max(0, int(np.random.normal(mean_N, std_N))),
                "P": max(0, int(np.random.normal(mean_P, std_P))),
                "K": max(0, int(np.random.normal(mean_K, std_K))),
                "temperature": round(np.random.normal(mean_temp, std_temp), 2),
                "humidity": round(np.clip(np.random.normal(mean_hum, std_hum), 0, 100), 2),
                "ph": round(np.clip(np.random.normal(mean_ph, std_ph), 0, 14), 2),
                "rainfall": round(max(0, np.random.normal(mean_rain, std_rain)), 2),
                "label": crop,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Crop_recommendation.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset generated: {output_path}  ({len(df)} rows, {df['label'].nunique()} crops)")
    return df


if __name__ == "__main__":
    generate_dataset()
