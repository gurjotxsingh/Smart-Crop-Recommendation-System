"""
InteliCrop – Streamlit Web Application
An interactive dashboard for crop prediction using ensemble ML models.

Usage:
    streamlit run app.py
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import load_data, preprocess_data, get_data_summary
from src.ensemble_model import (
    get_individual_models,
    train_and_evaluate,
    build_voting_ensemble,
    build_stacking_ensemble,
    run_full_pipeline,
    save_model,
    load_model,
    predict_crop,
)
from src.visualizations import (
    plot_model_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_correlation_heatmap,
    plot_crop_distribution,
    plot_probability_chart,
)
from src.generate_dataset import generate_dataset


# ── Page Configuration ───────────────────────────────────────────────────

st.set_page_config(
    page_title="InteliCrop – Smart Crop Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-card .label {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* Prediction result */
    .prediction-result {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.4);
    }
    .prediction-result h2 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }
    .prediction-result p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.15);
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────

CROP_EMOJI = {
    "rice": "🍚", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫛", "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘",
    "lentil": "🍲", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍏",
    "orange": "🍊", "papaya": "🫒", "coconut": "🥥", "cotton": "🧵",
    "jute": "🌿", "coffee": "☕",
}


@st.cache_data(show_spinner=False)
def cached_load_data():
    """Load and cache the dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "Crop_recommendation.csv")
    if not os.path.exists(data_path):
        generate_dataset(data_path)
    return load_data(data_path)


@st.cache_resource(show_spinner=False)
def cached_train_models():
    """Train all models and cache results."""
    df = cached_load_data()
    processed = preprocess_data(df, test_size=0.2, random_state=42)
    results = run_full_pipeline(
        processed["X_train"], processed["X_test"],
        processed["y_train"], processed["y_test"],
        label_encoder=processed["label_encoder"],
    )
    # Save best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    save_model(
        results[best_name]["model"],
        processed["scaler"],
        processed["label_encoder"],
    )
    return results, processed


@st.cache_resource(show_spinner=False)
def cached_load_model():
    """Load saved model bundle."""
    return load_model()


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌾 InteliCrop")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔬 Predict Crop", "📊 Model Analysis", "📈 Dataset Explorer"],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:rgba(255,255,255,0.5); font-size:0.8rem;'>
            <p>InteliCrop v1.0</p>
            <p>Ensemble ML for Smart Agriculture</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page: Home ───────────────────────────────────────────────────────────

if page == "🏠 Home":
    # Header
    st.markdown(
        """
        <div class='main-header'>
            <h1>🌾 InteliCrop</h1>
            <p>An Ensemble Machine Learning Model for Intelligent Crop Prediction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Description
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            ### What is InteliCrop?
            
            **InteliCrop** is an intelligent crop recommendation system that leverages 
            **ensemble machine learning** to predict the most suitable crop based on 
            soil nutrients and weather conditions.
            
            The system combines **7 individual classifiers** with **Voting** and 
            **Stacking** ensemble strategies to achieve maximum prediction accuracy.
            
            #### 🧬 Models Used
            | # | Algorithm | Type |
            |---|-----------|------|
            | 1 | Random Forest | Ensemble (Bagging) |
            | 2 | Gradient Boosting | Ensemble (Boosting) |
            | 3 | Extra Trees | Ensemble (Bagging) |
            | 4 | Decision Tree | Tree-based |
            | 5 | K-Nearest Neighbors | Instance-based |
            | 6 | SVM (RBF Kernel) | Kernel-based |
            | 7 | Naive Bayes | Probabilistic |
            | 8 | **Voting Ensemble** | **Meta-Ensemble** |
            | 9 | **Stacking Ensemble** | **Meta-Ensemble** |
            """,
        )

    with col2:
        st.markdown(
            """
            #### 🌍 Input Features
            
            <div class='info-box'>
                <strong>Soil Parameters</strong><br>
                • Nitrogen (N)<br>
                • Phosphorus (P)<br>
                • Potassium (K)<br>
                • Soil pH
            </div>
            <br>
            <div class='info-box'>
                <strong>Weather Parameters</strong><br>
                • Temperature (°C)<br>
                • Humidity (%)<br>
                • Rainfall (mm)
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick stats
    st.markdown("<p class='section-header'>📊 Quick Stats</p>", unsafe_allow_html=True)

    with st.spinner("Loading dataset..."):
        df = cached_load_data()
        summary = get_data_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='value'>{summary['shape'][0]:,}</div>
                <div class='label'>Total Samples</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='value'>{summary['num_classes']}</div>
                <div class='label'>Crop Types</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='value'>{len(summary['features'])}</div>
                <div class='label'>Features</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='value'>9</div>
                <div class='label'>ML Models</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        ### 🚀 Getting Started
        1. **Predict Crop** → Enter soil & weather data to get crop recommendations
        2. **Model Analysis** → Compare all 9 models and see detailed metrics
        3. **Dataset Explorer** → Explore the underlying dataset visually
        """
    )


# ── Page: Predict Crop ──────────────────────────────────────────────────

elif page == "🔬 Predict Crop":
    st.markdown(
        """
        <div class='main-header'>
            <h1>🔬 Crop Prediction</h1>
            <p>Enter soil and weather conditions to get intelligent crop recommendations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Train model if not already done
    model_path = os.path.join(os.path.dirname(__file__), "models", "intelicrop_model.pkl")
    if not os.path.exists(model_path):
        with st.spinner("🧠 Training ensemble models for first time... This may take a minute."):
            cached_train_models()

    # Input form
    st.markdown("<p class='section-header'>📝 Input Parameters</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧪 Soil Nutrients")
        nitrogen = st.slider("Nitrogen (N)", 0, 200, 50, help="Ratio of Nitrogen content in soil")
        phosphorus = st.slider("Phosphorus (P)", 0, 200, 50, help="Ratio of Phosphorus content in soil")
        potassium = st.slider("Potassium (K)", 0, 300, 50, help="Ratio of Potassium content in soil")
        ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1, help="pH value of the soil")

    with col2:
        st.markdown("#### 🌦️ Weather Conditions")
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5,
                                help="Average temperature in degree Celsius")
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0, 1.0,
                             help="Relative humidity in percentage")
        rainfall = st.slider("Rainfall (mm)", 0.0, 400.0, 100.0, 5.0,
                             help="Rainfall in millimeters")

    st.markdown("---")

    # Predict button
    if st.button("🌱 Predict Best Crop", use_container_width=True, type="primary"):
        with st.spinner("🔮 Analyzing conditions..."):
            try:
                bundle = cached_load_model()
                result = predict_crop(
                    bundle, nitrogen, phosphorus, potassium,
                    temperature, humidity, ph, rainfall,
                )

                crop = result["predicted_crop"]
                emoji = CROP_EMOJI.get(crop, "🌱")

                st.markdown(
                    f"""
                    <div class='prediction-result'>
                        <h2>{emoji} {crop.upper()}</h2>
                        <p>Recommended crop for the given conditions</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show confidence chart
                if "probabilities" in result:
                    st.markdown("<p class='section-header'>📊 Prediction Confidence</p>",
                                unsafe_allow_html=True)
                    fig = plot_probability_chart(result["probabilities"], top_n=5)
                    st.plotly_chart(fig, use_container_width=True)

                    # Show top 3 as cards
                    st.markdown("<p class='section-header'>🏆 Top 3 Recommendations</p>",
                                unsafe_allow_html=True)
                    top3 = list(result["probabilities"].items())[:3]
                    cols = st.columns(3)
                    for i, (crop_name, prob) in enumerate(top3):
                        with cols[i]:
                            e = CROP_EMOJI.get(crop_name, "🌱")
                            rank_color = ["#2ecc71", "#3498db", "#9b59b6"][i]
                            st.markdown(
                                f"""<div class='metric-card' style='border-top: 3px solid {rank_color};'>
                                    <div style='font-size:2rem;'>{e}</div>
                                    <div class='value' style='font-size:1.3rem;'>{crop_name.capitalize()}</div>
                                    <div class='label'>{prob*100:.1f}% confidence</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

            except FileNotFoundError:
                st.error("⚠️ Model not trained yet. Please wait while training completes...")
                with st.spinner("Training models..."):
                    cached_train_models()
                st.rerun()

    # Show input summary
    with st.expander("📋 Input Summary"):
        input_df = pd.DataFrame({
            "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
                          "Temperature", "Humidity", "pH", "Rainfall"],
            "Value": [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
            "Unit": ["ppm", "ppm", "ppm", "°C", "%", "", "mm"],
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)


# ── Page: Model Analysis ────────────────────────────────────────────────

elif page == "📊 Model Analysis":
    st.markdown(
        """
        <div class='main-header'>
            <h1>📊 Model Analysis</h1>
            <p>Compare ensemble and individual model performance across all metrics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("🧠 Training & evaluating all models..."):
        results, processed = cached_train_models()

    # Model Accuracy Comparison
    st.markdown("<p class='section-header'>🏆 Model Accuracy Comparison</p>", unsafe_allow_html=True)

    fig_comparison = plot_model_comparison(results)
    st.pyplot(fig_comparison)

    # Accuracy metrics
    st.markdown("<p class='section-header'>📈 Detailed Metrics</p>", unsafe_allow_html=True)

    sorted_models = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    # Best model highlight
    best_name, best_data = sorted_models[0]
    st.success(f"🏆 **Best Model: {best_name}** — Accuracy: **{best_data['accuracy']*100:.2f}%**")

    # Metrics table
    metrics_df = pd.DataFrame({
        "Model": [name for name, _ in sorted_models],
        "Accuracy (%)": [f"{data['accuracy']*100:.2f}" for _, data in sorted_models],
        "Rank": list(range(1, len(sorted_models) + 1)),
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Detailed per-model analysis
    st.markdown("<p class='section-header'>🔍 Per-Model Analysis</p>", unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Select a model to inspect",
        [name for name, _ in sorted_models],
    )

    if selected_model:
        model_data = results[selected_model]
        le = processed["label_encoder"]

        tab1, tab2, tab3 = st.tabs(["📋 Classification Report", "🔥 Confusion Matrix", "📊 Feature Importance"])

        with tab1:
            st.code(model_data["report"], language="text")

        with tab2:
            fig_cm = plot_confusion_matrix(
                model_data["confusion_matrix"],
                list(le.classes_),
                selected_model,
            )
            st.pyplot(fig_cm)

        with tab3:
            model_obj = model_data["model"]
            fig_fi = plot_feature_importance(
                model_obj, processed["feature_names"], selected_model
            )
            if fig_fi:
                st.pyplot(fig_fi)
            else:
                st.info(f"Feature importance is not available for {selected_model}.")


# ── Page: Dataset Explorer ──────────────────────────────────────────────

elif page == "📈 Dataset Explorer":
    st.markdown(
        """
        <div class='main-header'>
            <h1>📈 Dataset Explorer</h1>
            <p>Explore the crop recommendation dataset with interactive visualizations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = cached_load_data()

    # Dataset overview
    st.markdown("<p class='section-header'>📊 Dataset Overview</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Features", len(df.columns) - 1)
    with c3:
        st.metric("Crop Types", df["label"].nunique())

    # Data preview
    with st.expander("📄 Preview Data (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    # Statistical summary
    with st.expander("📈 Statistical Summary", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌾 Crop Distribution",
        "🔗 Correlation Matrix",
        "📊 Feature Distributions",
        "🔍 Crop-wise Analysis",
    ])

    with tab1:
        fig_dist = plot_crop_distribution(df)
        st.pyplot(fig_dist)

    with tab2:
        fig_corr = plot_correlation_heatmap(df)
        st.pyplot(fig_corr)

    with tab3:
        feature = st.selectbox("Select Feature", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        import plotly.express as px
        fig = px.histogram(
            df, x=feature, color="label",
            marginal="box", nbins=50,
            title=f"Distribution of {feature}",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        selected_crop = st.selectbox("Select Crop", sorted(df["label"].unique()))
        crop_df = df[df["label"] == selected_crop]

        st.markdown(f"**{selected_crop.capitalize()}** — {len(crop_df)} samples")
        st.dataframe(crop_df.describe(), use_container_width=True)

        # Radar chart for crop profile
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        means = crop_df[features].mean()
        overall_means = df[features].mean()

        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=means.values / overall_means.values,  # Normalize
            theta=features,
            fill="toself",
            name=selected_crop.capitalize(),
            line=dict(color="#667eea"),
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            title=f"Nutrient & Weather Profile – {selected_crop.capitalize()}",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)
