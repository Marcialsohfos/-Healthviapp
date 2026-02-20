"""
Page Tableau de bord - Visualisation des indicateurs clés
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import VULNERABILITY_COLORS, CLUSTER_COLORS, CLUSTER_LABELS
from utils.helpers import display_dataframe, create_download_link

st.set_page_config(page_title="Tableau de bord", page_icon="📊", layout="wide")

st.title("📊 Tableau de bord - Vulnérabilité sanitaire")

# Vérifier les données
if 'df' not in st.session_state or not st.session_state.data_loaded:
    st.warning("⚠️ Veuillez charger des données dans la page d'accueil")
    st.stop()

df = st.session_state.df

# Calculer l'IVS si nécessaire
if 'ivs_normalise' not in df.columns:
    st.warning("⚠️ Calcul de l'IVS en cours...")
    # Calcul simplifié
    df['ivs_normalise'] = 0.5  # À remplacer par le vrai calcul

# Filtres
st.sidebar.header("🔍 Filtres")

# Filtre par ville
if 'ville' in df.columns:
    villes = ['Toutes'] + list(df['ville'].unique())
    ville_filter = st.sidebar.selectbox("Ville", villes)
    if ville_filter != 'Toutes':
        df_filtered = df[df['ville'] == ville_filter].copy()
    else:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

# Filtre par niveau
if 'niveau_vulnerabilite' in df.columns:
    niveaux = ['Tous'] + list(df['niveau_vulnerabilite'].unique())
    niveau_filter = st.sidebar.selectbox("Niveau de vulnérabilité", niveaux)
    if niveau_filter != 'Tous':
        df_filtered = df_filtered[df_filtered['niveau_vulnerabilite'] == niveau_filter]

# Filtre par cluster
if 'Cluster' in df.columns:
    clusters = ['Tous'] + sorted(df['Cluster'].unique())
    cluster_filter = st.sidebar.selectbox("Cluster", clusters)
    if cluster_filter != 'Tous':
        df_filtered = df_filtered[df_filtered['Cluster'] == cluster_filter]

# KPIs
st.subheader("📌 Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total poches",
        len(df_filtered),
        delta=f"{len(df_filtered)-len(df):+d}" if len(df_filtered) != len(df) else None
    )

with col2:
    if 'ivs_normalise' in df_filtered.columns:
        ivs_moyen = df_filtered['ivs_normalise'].mean() * 100
        st.metric("IVS moyen", f"{ivs_moyen:.1f}%")

with col3:
    if 'ville' in df_filtered.columns:
        nb_villes = df_filtered['ville'].nunique()
        st.metric("Villes", nb_villes)

with col4:
    if 'quartier' in df_filtered.columns:
        nb_quartiers = df_filtered['quartier'].nunique()
        st.metric("Quartiers", nb_quartiers)

st.markdown("---")

# Graphiques
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Distribution par niveau")
    
    if 'niveau_vulnerabilite' in df_filtered.columns:
        level_counts = df_filtered['niveau_vulnerabilite'].value_counts().reset_index()
        level_counts.columns = ['Niveau', 'Nombre']
        
        colors = [VULNERABILITY_COLORS.get(lvl, '#95a5a6') for lvl in level_counts['Niveau']]
        
        fig = px.bar(
            level_counts,
            x='Niveau',
            y='Nombre',
            color='Niveau',
            color_discrete_sequence=colors,
            title="Répartition par niveau de vulnérabilité"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Colonne 'niveau_vulnerabilite' non disponible")

with col_right:
    st.subheader("🔴 Distribution par cluster")
    
    if 'Cluster' in df_filtered.columns:
        cluster_counts = df_filtered['Cluster'].value_counts().sort_index().reset_index()
        cluster_counts.columns = ['Cluster', 'Nombre']
        
        cluster_counts['Libellé'] = cluster_counts['Cluster'].map(CLUSTER_LABELS)
        
        colors = [CLUSTER_COLORS.get(c, '#95a5a6') for c in cluster_counts['Cluster']]
        
        fig = px.pie(
            cluster_counts,
            values='Nombre',
            names='Libellé',
            color_discrete_sequence=colors,
            title="Répartition par cluster"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Colonne 'Cluster' non disponible")

# Carte des clusters (simulée)
st.subheader("🗺️ Répartition spatiale par ville")

col_map1, col_map2 = st.columns(2)

with col_map1:
    st.markdown("**📍 Douala**")
    if 'ville' in df_filtered.columns:
        douala_data = df_filtered[df_filtered['ville'].str.contains('Douala', case=False, na=False)]
        if len(douala_data) > 0:
            cluster_dist = douala_data['Cluster'].value_counts().sort_index()
            
            fig = go.Figure(data=[go.Bar(
                x=[f"C{c}" for c in cluster_dist.index],
                y=cluster_dist.values,
                marker_color=[CLUSTER_COLORS.get(c, '#95a5a6') for c in cluster_dist.index],
                text=cluster_dist.values,
                textposition='auto'
            )])
            fig.update_layout(
                title=f"Douala - {len(douala_data)} poches",
                xaxis_title="Cluster",
                yaxis_title="Nombre"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée Douala")
    else:
        st.info("Colonne 'ville' non disponible")

with col_map2:
    st.markdown("**📍 Yaoundé**")
    if 'ville' in df_filtered.columns:
        yaounde_data = df_filtered[df_filtered['ville'].str.contains('Yaounde|Yaoundé', case=False, na=False)]
        if len(yaounde_data) > 0:
            cluster_dist = yaounde_data['Cluster'].value_counts().sort_index()
            
            fig = go.Figure(data=[go.Bar(
                x=[f"C{c}" for c in cluster_dist.index],
                y=cluster_dist.values,
                marker_color=[CLUSTER_COLORS.get(c, '#95a5a6') for c in cluster_dist.index],
                text=cluster_dist.values,
                textposition='auto'
            )])
            fig.update_layout(
                title=f"Yaoundé - {len(yaounde_data)} poches",
                xaxis_title="Cluster",
                yaxis_title="Nombre"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée Yaoundé")
    else:
        st.info("Colonne 'ville' non disponible")

# Tableau des données filtrées
st.subheader("📋 Données filtrées")
display_dataframe(df_filtered)

# Export
st.markdown(create_download_link(
    df_filtered,
    "donnees_filtrees.csv",
    "📥 Télécharger les données filtrées"
), unsafe_allow_html=True)