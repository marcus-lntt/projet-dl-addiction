"""Dashboard Streamlit — TikTok & Instagram Addiction Predictor (Jalon 9)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from src.models_dl import MLP
from src.preprocessing import FEATURES, TARGET, load_data

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/tiktok_instagram_global_100countries.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")


@st.cache_data
def load_df():
    return load_data(DATA_PATH)


@st.cache_resource
def load_models():
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    arch_path = os.path.join(MODELS_DIR, "mlp_arch.json")
    mlp_path = os.path.join(MODELS_DIR, "mlp_optimise.pt")
    ml_path = os.path.join(MODELS_DIR, "baseline_ridge.joblib")
    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")

    models = {}
    if os.path.exists(scaler_path):
        models["scaler"] = joblib.load(scaler_path)
    if os.path.exists(le_path):
        models["le"] = joblib.load(le_path)
    if os.path.exists(ml_path):
        models["ml"] = joblib.load(ml_path)
    if os.path.exists(arch_path) and os.path.exists(mlp_path):
        with open(arch_path) as f:
            arch = json.load(f)
        mlp = MLP(arch["input_dim"], tuple(arch["hidden_dims"]), arch["dropout"])
        mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))
        mlp.eval()
        models["mlp"] = mlp
    return models


# ── Load data & models ───────────────────────────────────────────────────────
try:
    df = load_df()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

models = load_models()
models_ready = "scaler" in models and ("ml" in models or "mlp" in models)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📱 Addiction Predictor")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "🔮 Prédiction", "🌍 Exploration mondiale", "📈 Tendances temporelles"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.title("📱 Social Media Addiction Score Predictor")
    st.markdown(
        """
        Ce dashboard permet d'explorer et de prédire le **score d'addiction aux réseaux sociaux**
        (TikTok & Instagram) à partir de données comportementales et démographiques.

        **Dataset** : *TikTok & Instagram Addiction Dataset (2015–2060)* — Kaggle
        **Licence** : CC BY-NC-SA 4.0

        ### Navigation
        | Page | Contenu |
        |---|---|
        | 🔮 Prédiction | Saisie de vos données et prédiction personnalisée |
        | 🌍 Exploration mondiale | Carte choroplèthe par pays |
        | 📈 Tendances temporelles | Évolution 2015–2060 |
        """
    )

    if not data_loaded:
        st.warning("⚠️ Dataset non chargé. Placez les fichiers CSV dans `data/`.")
    elif not models_ready:
        st.warning("⚠️ Modèles non trouvés. Exécutez d'abord les notebooks 02 et 03.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Observations", f"{len(df):,}")
        col2.metric("Pays", df["country"].nunique())
        col3.metric("Années", f"{df['year'].min()}–{df['year'].max()}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction du score d'addiction")

    if not models_ready:
        st.error("Modèles non disponibles. Exécutez les notebooks 02 et 03 en premier.")
        st.stop()

    scaler = models["scaler"]
    le = models.get("le")

    with st.form("prediction_form"):
        st.subheader("Profil utilisateur")
        col1, col2, col3 = st.columns(3)

        with col1:
            country = st.selectbox("Pays", df["country"].unique() if data_loaded else ["France"])
            age = st.slider("Âge", 10, 80, 25)
            year = st.slider("Année", 2015, 2060, 2024)

        with col2:
            tiktok_min = st.slider("TikTok (min/jour)", 0, 600, 90)
            insta_min = st.slider("Instagram (min/jour)", 0, 600, 60)
            sleep_h = st.slider("Sommeil (heures/nuit)", 3.0, 12.0, 7.5, step=0.5)

        with col3:
            attention = st.slider("Attention span score (0–100)", 0.0, 100.0, 70.0)
            dopamine = st.slider("Dopamine dependency (0–100)", 0.0, 100.0, 30.0)
            anxiety = st.slider("ASI — Anxiety Score Index (0–100)", 0.0, 100.0, 40.0)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        # Construire le vecteur de features dans l'ordre de FEATURES
        country_enc = le.transform([country])[0] if le is not None else 0

        # Valeurs moyennes du dataset pour les features non saisies
        if data_loaded:
            row_mean = df[df["country"] == country].mean(numeric_only=True)
        else:
            row_mean = pd.Series(dtype=float)

        def _get(col, default):
            return row_mean.get(col, default)

        feat_values = {
            "age": age,
            "year": year,
            "internet_penetration": _get("internet_penetration", 70.0),
            "mental_health_support_index": _get("mental_health_support_index", 60.0),
            "gdp_index": _get("gdp_index", 0.5),
            "youth_population_ratio": _get("youth_population_ratio", 0.3),
            "baseline_addiction_pressure": _get("baseline_addiction_pressure", 50.0),
            "tiktok_minutes_daily": tiktok_min,
            "instagram_minutes_daily": insta_min,
            "night_usage_ratio": _get("night_usage_ratio", 0.3),
            "scroll_velocity": _get("scroll_velocity", 3.0),
            "attention_span_score": attention,
            "dopamine_dependency_score": dopamine,
            "impulsivity_index": _get("impulsivity_index", 50.0),
            "sleep_hours": sleep_h,
            "sleep_quality_index": _get("sleep_quality_index", 6.0),
            "ASI": anxiety,
            "MHRI": _get("MHRI", 60.0),
            "country_enc": country_enc,
        }
        X_input = np.array([[feat_values[f] for f in FEATURES]], dtype=np.float32)
        X_scaled = scaler.transform(X_input)

        results = {}
        if "ml" in models:
            results["Ridge (ML)"] = float(models["ml"].predict(X_scaled)[0])
        if "mlp" in models:
            with torch.no_grad():
                results["MLP (DL)"] = float(
                    models["mlp"](torch.tensor(X_scaled, dtype=torch.float32)).item()
                )

        st.subheader("Résultats")
        cols = st.columns(len(results))
        for col, (name, score) in zip(cols, results.items()):
            score_clipped = max(0.0, min(100.0, score))
            label = "Faible" if score_clipped < 40 else ("Modéré" if score_clipped < 65 else "Élevé")
            col.metric(f"{name}", f"{score_clipped:.1f} / 100", f"Niveau : {label}")

        # Jauge
        avg_score = np.mean(list(results.values()))
        avg_score = max(0.0, min(100.0, avg_score))
        st.progress(avg_score / 100, text=f"Score moyen : {avg_score:.1f}/100")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : EXPLORATION MONDIALE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Exploration mondiale":
    st.title("🌍 Carte mondiale — Addiction Score par pays")

    if not data_loaded:
        st.error("Dataset non disponible.")
        st.stop()

    year_sel = st.slider("Année", int(df["year"].min()), int(df["year"].max()), 2024)
    metric_sel = st.selectbox(
        "Métrique",
        ["addiction_score", "tiktok_minutes_daily", "instagram_minutes_daily", "attention_span_score"],
    )

    df_year = df[df["year"] == year_sel].groupby("country")[metric_sel].mean().reset_index()

    fig = px.choropleth(
        df_year,
        locations="country",
        locationmode="country names",
        color=metric_sel,
        color_continuous_scale="RdYlGn_r",
        title=f"{metric_sel} moyen par pays ({year_sel})",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Top 10 pays — {metric_sel} ({year_sel})")
    top10 = df_year.sort_values(metric_sel, ascending=False).head(10)
    st.dataframe(top10.style.background_gradient(subset=[metric_sel], cmap="Reds"), hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : TENDANCES TEMPORELLES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Tendances temporelles":
    st.title("📈 Évolution temporelle 2015–2060")

    if not data_loaded:
        st.error("Dataset non disponible.")
        st.stop()

    countries_sel = st.multiselect(
        "Pays à comparer",
        df["country"].unique().tolist(),
        default=["France", "United States", "Japan"] if all(c in df["country"].values for c in ["France", "United States", "Japan"]) else df["country"].unique()[:3].tolist(),
    )
    metric_ts = st.selectbox(
        "Métrique",
        ["addiction_score", "attention_span_score", "tiktok_minutes_daily", "instagram_minutes_daily"],
    )

    if countries_sel:
        df_ts = df[df["country"].isin(countries_sel)].groupby(["year", "country"])[metric_ts].mean().reset_index()
        fig = px.line(
            df_ts,
            x="year",
            y=metric_ts,
            color="country",
            markers=True,
            title=f"Évolution de {metric_ts} par pays (2015–2060)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tendance mondiale
    st.subheader("Tendance mondiale (moyenne tous pays)")
    world = df.groupby("year")[metric_ts].mean().reset_index()
    fig2 = px.area(world, x="year", y=metric_ts, title=f"Tendance mondiale — {metric_ts}", height=300)
    st.plotly_chart(fig2, use_container_width=True)
