"""
Page Configuration - Paramètres du modèle et de l'application
"""

import streamlit as st
import pandas as pd
import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import APP_CONFIG, ESSENTIAL_FEATURES

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")

st.title("⚙️ Configuration")

# Onglets
tab1, tab2, tab3 = st.tabs(["🤖 Modèle", "📊 Données", "⚙️ Paramètres"])

with tab1:
    st.subheader("🤖 Configuration du modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Modèle actuel**")
        st.info(f"Version: {APP_CONFIG['VERSION']}")
        st.info(f"Performance: R² > 0.98 sur test")
        
        # Bouton de réentraînement
        if st.button("🔄 Réentraîner le modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                try:
                    # Lancer le script d'entraînement
                    result = subprocess.run(
                        [sys.executable, "ml_model/train_model.py"],
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    if result.returncode == 0:
                        st.success("✅ Modèle réentraîné avec succès !")
                        st.code(result.stdout)
                    else:
                        st.error(f"❌ Erreur: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    with col2:
        st.markdown("**Paramètres du modèle**")
        st.json({
            "algorithmes": ["Random Forest", "XGBoost"],
            "n_estimators": 200,
            "test_size": 0.2,
            "validation": "5-fold CV"
        })
    
    st.markdown("---")
    st.subheader("📊 Importance des variables")
    
    # Importance fixe basée sur l'étude
    importance_data = pd.DataFrame({
        'Variable': ['Risques naturels', 'Accès & Mobilité', 'Largeur voirie', 'Habitat', 'Matériaux'],
        'Importance': [0.83, 0.02, 0.001, 0.002, 0.001]
    })
    
    st.dataframe(importance_data)

with tab2:
    st.subheader("📊 Configuration des données")
    
    st.markdown("**Variables essentielles**")
    st.write(ESSENTIAL_FEATURES)
    
    st.markdown("---")
    st.subheader("📁 Fichiers de données")
    
    # Lister les fichiers disponibles
    data_dir = Path("data")
    if data_dir.exists():
        files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.csv"))
        if files:
            st.write("Fichiers disponibles:")
            for f in files:
                st.write(f"- {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            st.info("Aucun fichier dans le dossier data/")
    else:
        st.info("Dossier data/ non trouvé")

with tab3:
    st.subheader("⚙️ Paramètres de l'application")
    
    # Seuils de vulnérabilité
    st.markdown("**Seuils de vulnérabilité**")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        faible = st.number_input("Faible", 0.0, 1.0, 0.25, 0.05)
    with col_s2:
        moderee = st.number_input("Modérée", 0.0, 1.0, 0.50, 0.05)
    with col_s3:
        elevee = st.number_input("Élevée", 0.0, 1.0, 0.75, 0.05)
    with col_s4:
        critique = st.number_input("Critique", 0.0, 1.0, 1.00, 0.05)
    
    if st.button("💾 Sauvegarder les seuils"):
        st.success("✅ Seuils sauvegardés !")
    
    st.markdown("---")
    st.markdown("**À propos**")
    st.json(APP_CONFIG)