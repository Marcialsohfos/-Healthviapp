"""
Page Analyses - Graphiques et statistiques avancées
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import VULNERABILITY_COLORS, CLUSTER_COLORS
from utils.helpers import display_dataframe, create_download_link

st.set_page_config(page_title="Analyses", page_icon="📈", layout="wide")

st.title("📈 Analyses approfondies")

# Vérifier les données
if 'df' not in st.session_state or not st.session_state.data_loaded:
    st.warning("⚠️ Veuillez charger des données dans la page d'accueil")
    st.stop()

df = st.session_state.df

# Calculer l'IVS si nécessaire
if 'ivs_normalise' not in df.columns:
    df['ivs_normalise'] = 0.5  # À remplacer

# Sidebar pour les options d'analyse
st.sidebar.header("🔬 Options d'analyse")

# Type d'analyse
analysis_type = st.sidebar.selectbox(
    "Type d'analyse",
    ["Distribution", "Corrélations", "Comparaison villes", "Importance des facteurs"]
)

if analysis_type == "Distribution":
    st.subheader("📊 Analyse des distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution de l'IVS
        fig = px.histogram(
            df,
            x='ivs_normalise',
            nbins=20,
            title="Distribution de l'IVS",
            labels={'ivs_normalise': 'IVS normalisé'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution par ville
        if 'ville' in df.columns:
            ville_counts = df['ville'].value_counts().reset_index()
            ville_counts.columns = ['Ville', 'Nombre']
            
            fig = px.pie(
                ville_counts,
                values='Nombre',
                names='Ville',
                title="Répartition par ville"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Distribution par cluster
    if 'Cluster' in df.columns:
        st.subheader("📊 Distribution par cluster")
        
        cluster_stats = df.groupby('Cluster').agg({
            'ivs_normalise': ['count', 'mean', 'std']
        }).round(3)
        cluster_stats.columns = ['Nombre', 'IVS moyen', 'Écart-type']
        
        st.dataframe(cluster_stats)

elif analysis_type == "Corrélations":
    st.subheader("🔗 Matrice de corrélation")
    
    # Sélectionner les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    selected_cols = st.multiselect(
        "Colonnes à inclure",
        numeric_cols,
        default=[c for c in numeric_cols if 'score' in c or 'ivs' in c][:8]
    )
    
    if len(selected_cols) >= 2:
        corr_matrix = df[selected_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matrice de corrélation",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sélectionnez au moins 2 colonnes")

elif analysis_type == "Comparaison villes":
    st.subheader("🏙️ Comparaison Douala - Yaoundé")
    
    if 'ville' in df.columns:
        douala = df[df['ville'].str.contains('Douala', case=False, na=False)]
        yaounde = df[df['ville'].str.contains('Yaounde|Yaoundé', case=False, na=False)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📍 Douala**")
            st.metric("Nombre de poches", len(douala))
            if 'ivs_normalise' in douala.columns:
                st.metric("IVS moyen", f"{douala['ivs_normalise'].mean()*100:.1f}%")
        
        with col2:
            st.markdown("**📍 Yaoundé**")
            st.metric("Nombre de poches", len(yaounde))
            if 'ivs_normalise' in yaounde.columns:
                st.metric("IVS moyen", f"{yaounde['ivs_normalise'].mean()*100:.1f}%")
        
        # Boxplot comparatif
        if 'ivs_normalise' in df.columns:
            fig = px.box(
                df,
                x='ville',
                y='ivs_normalise',
                title="Comparaison des IVS par ville",
                labels={'ivs_normalise': 'IVS', 'ville': 'Ville'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top 5 par ville
        st.subheader("🏆 Top 5 des poches les plus vulnérables")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("**Douala**")
            if len(douala) > 0:
                top_douala = douala.nlargest(5, 'ivs_normalise')[['quartier', 'ivs_normalise']]
                st.dataframe(top_douala)
        
        with col_t2:
            st.markdown("**Yaoundé**")
            if len(yaounde) > 0:
                top_yaounde = yaounde.nlargest(5, 'ivs_normalise')[['quartier', 'ivs_normalise']]
                st.dataframe(top_yaounde)

elif analysis_type == "Importance des facteurs":
    st.subheader("🎯 Importance des facteurs de vulnérabilité")
    
    # Utiliser l'importance du modèle
    from ml_model.predict import get_predictor
    predictor = get_predictor()
    
    importance = predictor.get_feature_importance(top_n=10)
    
    if importance is not None:
        fig = px.bar(
            importance,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 des facteurs les plus importants",
            labels={'importance': 'Importance', 'feature': 'Facteur'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Importance des facteurs non disponible")