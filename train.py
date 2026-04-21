"""
InteliCrop - Model Training Script
Run this script to train all models, evaluate them, and save the best one.

Usage:
    python train.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.generate_dataset import generate_dataset
from src.data_preprocessing import load_data, preprocess_data
from src.ensemble_model import run_full_pipeline, save_model


def main():
    """Main training entry point."""
    print("\nInteliCrop - Training Pipeline\n")

    # --- Step 1: Ensure dataset exists ---
    data_path = os.path.join(os.path.dirname(__file__), "data", "Crop_recommendation.csv")
    if not os.path.exists(data_path):
        print("[!] Dataset not found. Generating synthetic dataset...")
        generate_dataset(data_path)
    else:
        print(f"[OK] Dataset found: {data_path}")

    # --- Step 2: Load & preprocess ---
    df = load_data(data_path)
    processed = preprocess_data(df, test_size=0.2, random_state=42)

    # --- Step 3: Train all models ---
    results = run_full_pipeline(
        processed["X_train"],
        processed["X_test"],
        processed["y_train"],
        processed["y_test"],
        label_encoder=processed["label_encoder"],
    )

    # --- Step 4: Save best model ---
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]

    print(f"\n[->] Saving best model: {best_name} (Accuracy: {best_acc:.4f})")
    save_model(
        model=best_model,
        scaler=processed["scaler"],
        label_encoder=processed["label_encoder"],
        filename="intelicrop_model.pkl",
    )

    # Also save individual results summary
    print("\nTraining complete! Run the app with: streamlit run app.py\n")


if __name__ == "__main__":
    main()
