"""
InteliCrop - Visualization Utilities
Helper functions for generating plots and charts.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_model_comparison(results: dict):
    """
    Create a horizontal bar chart comparing model accuracies.

    Parameters
    ----------
    results : dict
        {model_name: {"accuracy": float, ...}}

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]

    # Sort by accuracy
    sorted_pairs = sorted(zip(accuracies, names), reverse=False)
    accuracies, names = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("viridis", len(names))
    bars = ax.barh(names, accuracies, color=colors, edgecolor="white", height=0.6)

    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2f}%", va="center", fontweight="bold", fontsize=10)

    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("InteliCrop – Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, model_name="Model"):
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : list
        List of class labels
    model_name : str
        Name of the model for the title

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, model_name="Model"):
    """
    Plot feature importances for tree-based models.

    Parameters
    ----------
    model : estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        Feature column names
    model_name : str
        Model name for the title

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("magma_r", len(feature_names))
    ax.barh([feature_names[i] for i in sorted_idx],
            importances[sorted_idx],
            color=[colors[i] for i in range(len(sorted_idx))],
            edgecolor="white", height=0.6)

    for i, idx in enumerate(sorted_idx):
        ax.text(importances[idx] + 0.005, i,
                f"{importances[idx]:.3f}", va="center", fontsize=10)

    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance – {model_name}", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """
    Plot a correlation matrix heatmap for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset

    Returns
    -------
    matplotlib.figure.Figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_crop_distribution(df):
    """
    Plot bar chart of crop class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'label' column

    Returns
    -------
    matplotlib.figure.Figure
    """
    crop_counts = df["label"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Spectral", len(crop_counts))
    bars = ax.barh(crop_counts.index, crop_counts.values, color=colors, edgecolor="white")

    for bar, count in zip(bars, crop_counts.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)

    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_title("Crop Distribution in Dataset", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_probability_chart(probabilities: dict, top_n: int = 5):
    """
    Create a Plotly bar chart for prediction probabilities.

    Parameters
    ----------
    probabilities : dict
        {crop_name: probability}
    top_n : int
        Number of top crops to display

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Get top N
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    crops = [p[0].capitalize() for p in sorted_probs]
    probs = [p[1] * 100 for p in sorted_probs]

    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(crops))]

    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=crops,
            orientation="h",
            marker=dict(color=colors, line=dict(width=1, color="white")),
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
        )
    ])

    fig.update_layout(
        title="Top Predicted Crops (Confidence %)",
        xaxis_title="Confidence (%)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=50, b=30),
    )

    return fig
