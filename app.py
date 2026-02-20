"""
Application Streamlit - IA Vulnérabilité Sanitaire
Page d'accueil
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.constants import APP_CONFIG, VULNERABILITY_COLORS
from utils.helpers import display_dataframe, create_download_link
from ml_model.predict import get_predictor

# Configuration de la page
st.set_page_config(
    page_title=APP_CONFIG['NAME'],
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de la session
if 'predictor' not in st.session_state:
    st.session_state.predictor = get_predictor()

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/health-organization.png")
    st.title(f"🏥 {APP_CONFIG['NAME']}")
    st.caption(f"Version {APP_CONFIG['VERSION']}")
    st.caption(f"© {APP_CONFIG['YEAR']} - {APP_CONFIG['AUTHOR']}")
    
    st.divider()
    
    st.subheader("📊 À propos")
    st.info(
        """
        Cette application utilise l'intelligence artificielle pour 
        prédire et analyser la vulnérabilité sanitaire dans les 
        quartiers précaires de Douala et Yaoundé.
        
        **Données**: 266 poches d'habitat précaire (2025)
        **Modèle**: Random Forest / XGBoost (R² > 0.98)
        """
    )
    
    st.divider()
    
    # Chargement des données
    st.subheader("📂 Chargement des données")
    uploaded_file = st.file_uploader(
        "Choisir un fichier Excel",
        type=['xlsx', 'xls', 'csv'],
        help="Format attendu: colonnes id_poche, ville, quartier, etc."
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ {len(df)} lignes chargées")
            
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
    
    st.divider()
    
    # Statistiques
    if st.session_state.data_loaded:
        st.subheader("📈 Statistiques")
        df = st.session_state.df
        st.metric("Nombre de poches", len(df))
        if 'ville' in df.columns:
            st.metric("Villes", df['ville'].nunique())

# Page principale
st.title("🏥 IA Vulnérabilité Sanitaire - Douala & Yaoundé")
st.markdown("---")

# Colonnes principales
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 266 poches d'habitat précaire")
    st.markdown(
        """
        - **Douala**: 95 poches
        - **Yaoundé**: 171 poches
        - Enquête MINHDU/BUCREP 2025
        """
    )

with col2:
    st.subheader("🤖 Modèle prédictif")
    st.markdown(
        """
        - **R²**: 0.98 (test)
        - **Algorithmes**: Random Forest, XGBoost
        - **Facteurs clés**: Risques naturels, Accès
        """
    )

with col3:
    st.subheader("🎯 Objectif")
    st.markdown(
        """
        Prioriser les interventions sanitaires
        par quartier et type de vulnérabilité
        """
    )

st.markdown("---")

# Aperçu des données
if st.session_state.data_loaded:
    st.subheader("📋 Aperçu des données")
    display_dataframe(st.session_state.df)
    
    # Téléchargement
    st.markdown(create_download_link(
        st.session_state.df, 
        "export_donnees.csv", 
        "📥 Télécharger les données"
    ), unsafe_allow_html=True)

else:
    # Message d'accueil
    st.info(
        """
        👈 Commencez par charger vos données dans le panneau latéral.
        
        **Fonctionnalités disponibles:**
        - **Tableau de bord**: Visualisation des indicateurs clés
        - **Prédiction**: Calcul de l'IVS pour de nouveaux quartiers
        - **Analyses**: Graphiques et statistiques détaillées
        - **Configuration**: Paramètres du modèle
        """
    )

# Footer
st.markdown("---")
st.caption(f"© {APP_CONFIG['YEAR']} - {APP_CONFIG['AUTHOR']} | Source: {APP_CONFIG['DATA_SOURCE']}")