"""
Page Prédiction - Calcul de l'IVS pour de nouveaux quartiers
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model.predict import get_predictor
from utils.constants import VULNERABILITY_COLORS, FACTEURS_CLES, RECOMMENDATIONS
from utils.helpers import create_progress_bar

st.set_page_config(page_title="Prédiction", page_icon="🤖", layout="wide")

st.title("🤖 Prédiction de la vulnérabilité sanitaire")

# Initialiser le prédicteur
predictor = get_predictor()

# Créer des onglets
tab1, tab2, tab3 = st.tabs(["📝 Formulaire", "📊 Prédiction unique", "📦 Prédiction batch"])

with tab1:
    st.subheader("📝 Saisie des caractéristiques du quartier")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📍 Localisation**")
        ville = st.selectbox("Ville", ["Douala", "Yaoundé"])
        quartier = st.text_input("Nom du quartier")
        
        st.markdown("**🏠 Habitat**")
        mur = st.selectbox(
            "Matériaux des murs",
            ["Terre/Banco", "Planche", "Brique terre", "Parpaing", "Béton"]
        )
        toit = st.selectbox(
            "Matériaux du toit",
            ["Chaume", "Tôle", "Bac alu", "Tuile", "Béton"]
        )
        densite = st.slider("Densité de logements (logements/ha)", 0, 500, 100)
    
    with col2:
        st.markdown("**🚰 Services de base**")
        eau = st.selectbox("Accès à l'eau", ["Oui", "Non", "Partiel"])
        evac_eau = st.selectbox("Évacuation des eaux", ["Réseau", "Fossé", "Nature", "Aucune"])
        elec = st.selectbox("Électricité", ["Oui", "Non", "Partiel"])
        
        st.markdown("**⚠️ Risques**")
        risque_nat = st.multiselect(
            "Risques naturels",
            ["Inondation", "Glissement", "Éboulement"]
        )
        risque_art = st.multiselect(
            "Risques artificiels",
            ["Haute tension", "Pollution", "Décharge"]
        )
    
    if st.button("🔮 Prédire", type="primary", use_container_width=True):
        # Construire le DataFrame d'entrée
        input_data = {
            'ville': ville,
            'quartier': quartier,
            'larg_voiri': 5.0,  # Valeur par défaut
            'mat_mur': mur,
            'mat_toit': toit,
            'dens_log': densite,
            'eau_bois': eau,
            'evac_eau': evac_eau,
            'elec': elec,
            'risq_nat': '|'.join(risque_nat) if risque_nat else 'Aucun'
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Faire la prédiction
        result = predictor.predict(input_df)
        
        if result['success']:
            st.session_state.prediction_result = result
            st.session_state.prediction_input = input_data
            st.success("✅ Prédiction effectuée avec succès !")
        else:
            st.error(f"❌ Erreur: {result.get('error', 'Inconnue')}")

with tab2:
    st.subheader("📊 Résultat de la prédiction")
    
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        input_data = st.session_state.prediction_input
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📈 Indice de Vulnérabilité Sanitaire (IVS)**")
            
            # Afficher la jauge
            ivs_pct = result['prediction_pct']
            niveau = result['niveau']
            couleur = result['couleur']
            
            st.markdown(create_progress_bar(ivs_pct), unsafe_allow_html=True)
            
            st.markdown(f"""
            **IVS**: {ivs_pct:.1f}%
            **Niveau**: :{couleur}[**{niveau}**]
            **Confiance**: {result['confidence']*100:.0f}%
            """)
            
            # Recommandations
            st.markdown("**📋 Recommandations**")
            for rec in RECOMMENDATIONS.get(niveau, ["Aucune recommandation"]):
                st.markdown(f"- {rec}")
        
        with col2:
            st.markdown("**📝 Données saisies**")
            st.json(input_data)
            
            # Facteurs clés
            st.markdown("**🔑 Facteurs clés potentiels**")
            risque_principal = "Inondation" if 'Inondation' in str(input_data['risq_nat']) else "Général"
            for facteur in FACTEURS_CLES.get(risque_principal, FACTEURS_CLES['Défaut_accès'])[:3]:
                st.markdown(f"- {facteur}")
        
        # Bouton d'export
        if st.button("📥 Exporter le résultat"):
            result_df = pd.DataFrame([{
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'quartier': input_data['quartier'],
                'ville': input_data['ville'],
                'ivs': result['prediction'],
                'ivs_pct': result['prediction_pct'],
                'niveau': result['niveau'],
                'confiance': result['confidence']
            }])
            csv = result_df.to_csv(index=False)
            st.download_button(
                "📥 Télécharger CSV",
                csv,
                "prediction.csv",
                "text/csv"
            )
    else:
        st.info("👈 Effectuez d'abord une prédiction dans l'onglet 'Formulaire'")

with tab3:
    st.subheader("📦 Prédiction par lot")
    
    uploaded_file = st.file_uploader(
        "Charger un fichier de plusieurs quartiers",
        type=['xlsx', 'xls', 'csv']
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ {len(batch_df)} lignes chargées")
            st.dataframe(batch_df.head())
            
            if st.button("🚀 Prédire tout le lot"):
                with st.spinner("Prédictions en cours..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, (_, row) in enumerate(batch_df.iterrows()):
                        row_df = pd.DataFrame([row])
                        pred = predictor.predict(row_df)
                        
                        if pred['success']:
                            results.append({
                                'index': i,
                                'ivs': pred['prediction'],
                                'ivs_pct': pred['prediction_pct'],
                                'niveau': pred['niveau']
                            })
                        
                        progress_bar.progress((i + 1) / len(batch_df))
                    
                    results_df = pd.DataFrame(results)
                    st.success(f"✅ {len(results_df)} prédictions réussies")
                    
                    # Fusionner avec les données originales
                    final_df = batch_df.reset_index().join(results_df.set_index('index'))
                    
                    st.dataframe(final_df)
                    
                    # Export
                    csv = final_df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger les résultats",
                        csv,
                        "predictions_batch.csv",
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Erreur: {e}")