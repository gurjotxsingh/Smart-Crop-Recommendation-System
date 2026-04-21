"""
InteliCrop - Ensemble Model Module
Implements individual ML classifiers and the final ensemble (Voting + Stacking).
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ── Individual Model Definitions ──────────────────────────────────────────

def get_individual_models() -> dict:
    """
    Return a dictionary of individual ML classifiers used in the ensemble.

    Returns
    -------
    dict
        {model_name: sklearn_estimator}
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale", probability=True, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
    }
    return models


# ── Training & Evaluation ────────────────────────────────────────────────

def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test, label_encoder=None):
    """
    Train each model and return evaluation results.

    Parameters
    ----------
    models : dict
        {model_name: estimator}
    X_train, X_test : array-like
        Scaled feature matrices
    y_train, y_test : array-like
        Encoded labels
    label_encoder : LabelEncoder, optional
        For decoding class names

    Returns
    -------
    dict
        {model_name: {"model": fitted_model, "accuracy": float, "report": str, "cm": array}}
    """
    results = {}
    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        target_names = list(label_encoder.classes_) if label_encoder else None
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
        }
        print(f"Accuracy: {acc:.4f}")

    return results


# ── Ensemble Models ──────────────────────────────────────────────────────

def build_voting_ensemble(models: dict, voting: str = "soft") -> VotingClassifier:
    """
    Build a Voting Classifier ensemble from individual models.

    Parameters
    ----------
    models : dict
        {model_name: estimator}
    voting : str
        'hard' or 'soft' (soft uses predicted probabilities)

    Returns
    -------
    VotingClassifier
    """
    estimators = [(name.lower().replace(" ", "_").replace("(", "").replace(")", ""), model)
                  for name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


def build_stacking_ensemble(models: dict) -> StackingClassifier:
    """
    Build a Stacking Classifier with Logistic Regression as the meta-learner.

    Parameters
    ----------
    models : dict
        {model_name: estimator}

    Returns
    -------
    StackingClassifier
    """
    estimators = [(name.lower().replace(" ", "_").replace("(", "").replace(")", ""), model)
                  for name, model in models.items()]
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
    )


# ── Full Pipeline ────────────────────────────────────────────────────────

def run_full_pipeline(X_train, X_test, y_train, y_test, label_encoder=None):
    """
    Execute the complete InteliCrop training pipeline.

    1. Train all individual models
    2. Build & train Voting Ensemble
    3. Build & train Stacking Ensemble
    4. Compare and select best model

    Parameters
    ----------
    X_train, X_test : array-like
    y_train, y_test : array-like
    label_encoder : LabelEncoder, optional

    Returns
    -------
    dict
        Complete results including individual + ensemble models
    """
    print("\n" + "=" * 60)
    print("  InteliCrop – Ensemble Model Training Pipeline")
    print("=" * 60)

    # Step 1: Individual models
    print("\n[1/3] Training Individual Models...")
    individual_models = get_individual_models()
    results = train_and_evaluate(
        individual_models, X_train, X_test, y_train, y_test, label_encoder
    )

    # Step 2: Voting Ensemble
    print("\n[2/3] Training Voting Ensemble (Soft)...")
    voting_clf = build_voting_ensemble(get_individual_models(), voting="soft")
    voting_clf.fit(X_train, y_train)
    y_pred_v = voting_clf.predict(X_test)
    acc_v = accuracy_score(y_test, y_pred_v)
    target_names = list(label_encoder.classes_) if label_encoder else None
    results["Voting Ensemble"] = {
        "model": voting_clf,
        "accuracy": acc_v,
        "report": classification_report(y_test, y_pred_v, target_names=target_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_v),
    }
    print(f"  Voting Ensemble Accuracy: {acc_v:.4f}")

    # Step 3: Stacking Ensemble
    print("\n[3/3] Training Stacking Ensemble...")
    stacking_clf = build_stacking_ensemble(get_individual_models())
    stacking_clf.fit(X_train, y_train)
    y_pred_s = stacking_clf.predict(X_test)
    acc_s = accuracy_score(y_test, y_pred_s)
    results["Stacking Ensemble"] = {
        "model": stacking_clf,
        "accuracy": acc_s,
        "report": classification_report(y_test, y_pred_s, target_names=target_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_s),
    }
    print(f"  Stacking Ensemble Accuracy: {acc_s:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("  Model Comparison Summary")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for rank, (name, res) in enumerate(sorted_results, 1):
        marker = " * BEST" if rank == 1 else ""
        print(f"  {rank}. {name:<25s} -> {res['accuracy']:.4f}{marker}")

    best_name = sorted_results[0][0]
    print(f"\n[OK] Best Model: {best_name} ({sorted_results[0][1]['accuracy']:.4f})")
    print("=" * 60)

    return results


# ── Save / Load ──────────────────────────────────────────────────────────

def save_model(model, scaler, label_encoder, model_dir: str = None, filename: str = "intelicrop_model.pkl"):
    """
    Save trained model, scaler, and label encoder to disk.

    Parameters
    ----------
    model : estimator
        Trained sklearn model
    scaler : StandardScaler
        Fitted scaler
    label_encoder : LabelEncoder
        Fitted label encoder
    model_dir : str, optional
        Directory to save in. Defaults to models/
    filename : str
        Filename for the saved model bundle
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }
    path = os.path.join(model_dir, filename)
    joblib.dump(bundle, path)
    print(f"[OK] Model saved: {path}")


def load_model(model_dir: str = None, filename: str = "intelicrop_model.pkl"):
    """
    Load a saved model bundle from disk.

    Parameters
    ----------
    model_dir : str, optional
        Directory containing the model. Defaults to models/
    filename : str
        Model bundle filename

    Returns
    -------
    dict
        {"model": estimator, "scaler": StandardScaler, "label_encoder": LabelEncoder}
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at {path}")
    bundle = joblib.load(path)
    print(f"[OK] Model loaded: {path}")
    return bundle


def predict_crop(model_bundle: dict, N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict crop for given soil and weather conditions.

    Parameters
    ----------
    model_bundle : dict
        Output of load_model()
    N, P, K : float
        Nitrogen, Phosphorus, Potassium levels
    temperature : float
        Temperature in °C
    humidity : float
        Relative humidity (%)
    ph : float
        Soil pH
    rainfall : float
        Rainfall in mm

    Returns
    -------
    dict
        {"predicted_crop": str, "probabilities": dict}
    """
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    le = model_bundle["label_encoder"]

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    crop_name = le.inverse_transform([prediction])[0]

    result = {"predicted_crop": crop_name}

    # Get probabilities if the model supports it
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(features_scaled)[0]
        prob_dict = {le.inverse_transform([i])[0]: round(float(p), 4)
                     for i, p in enumerate(probas)}
        # Sort by probability descending
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        result["probabilities"] = prob_dict

    return result
