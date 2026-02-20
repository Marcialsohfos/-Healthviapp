"""
Fonctions utilitaires pour l'application Streamlit
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import io
import base64

def display_dataframe(df, max_rows=50):
    """
    Affiche un dataframe avec style
    """
    if df is None or df.empty:
        st.warning("Aucune donnée à afficher")
        return
    
    # Limiter le nombre de lignes
    display_df = df.head(max_rows)
    
    # Afficher avec style
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "icv_normalise": st.column_config.ProgressColumn(
                "ICV",
                help="Indice de Vulnérabilité Composite",
                format="%.1f",
                min_value=0,
                max_value=100,
            )
        } if "icv_normalise" in display_df.columns else None
    )
    
    # Afficher les informations
    st.caption(f"Affichage de {len(display_df)} sur {len(df)} lignes")

def create_download_link(df, filename="export.csv", text="📥 Télécharger"):
    """
    Crée un lien de téléchargement pour un DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_color_for_level(level):
    """
    Retourne la couleur associée à un niveau de vulnérabilité
    """
    colors = {
        'Faible': '#27ae60',
        'Modérée': '#f39c12',
        'Élevée': '#e67e22',
        'Critique': '#e74c3c'
    }
    return colors.get(level, '#95a5a6')

def create_progress_bar(value, max_value=100):
    """
    Crée une barre de progression avec couleur
    """
    if value <= 25:
        color = 'green'
    elif value <= 50:
        color = 'yellow'
    elif value <= 75:
        color = 'orange'
    else:
        color = 'red'
    
    progress_html = f"""
    <div style="background-color: #ecf0f1; border-radius: 10px; padding: 3px;">
        <div style="background-color: {color}; 
                    width: {value}%; 
                    height: 20px; 
                    border-radius: 8px;
                    text-align: center;
                    color: white;
                    font-weight: bold;">
            {value:.1f}%
        </div>
    </div>
    """
    return progress_html

def format_number(value):
    """
    Formate un nombre pour l'affichage
    """
    if pd.isna(value):
        return "-"
    
    if isinstance(value, (int, np.integer)):
        return f"{value:,}".replace(",", " ")
    elif isinstance(value, (float, np.floating)):
        if value == 0:
            return "0"
        elif abs(value) < 0.01:
            return f"{value:.2e}"
        elif abs(value) < 1:
            return f"{value:.3f}"
        elif abs(value) < 1000:
            return f"{value:.1f}"
        else:
            return f"{value:,.0f}".replace(",", " ")
    else:
        return str(value)

def calculate_statistics(df):
    """
    Calcule des statistiques descriptives
    """
    stats = {}
    
    if df is None or df.empty:
        return stats
    
    # Pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols[:10]:  # Limiter aux 10 premières
        stats[col] = {
            'moyenne': df[col].mean(),
            'médiane': df[col].median(),
            'écart-type': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'non_nulls': df[col].count()
        }
    
    return stats

def filter_dataframe(df, filters):
    """
    Filtre un dataframe selon des critères
    """
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    for column, value in filters.items():
        if column in filtered_df.columns and value:
            if isinstance(value, (list, tuple)):
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df

def create_summary_cards(df, column='icv_normalise'):
    """
    Crée des cartes de résumé
    """
    if df is None or df.empty or column not in df.columns:
        return {}
    
    return {
        'total': len(df),
        'moyenne': df[column].mean(),
        'médiane': df[column].median(),
        'min': df[column].min(),
        'max': df[column].max(),
        'écart_type': df[column].std()
    }

def validate_file_upload(file):
    """
    Valide un fichier uploadé
    """
    if file is None:
        return False, "Aucun fichier sélectionné"
    
    # Vérifier l'extension
    allowed_extensions = ['.xlsx', '.xls', '.csv']
    file_extension = Path(file.name).suffix.lower()
    
    if file_extension not in allowed_extensions:
        return False, f"Format non supporté. Formats autorisés: {', '.join(allowed_extensions)}"
    
    # Vérifier la taille (max 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    if file.size > max_size:
        return False, f"Fichier trop volumineux. Maximum: {max_size/1024/1024:.0f}MB"
    
    return True, "Fichier valide"

@st.cache_data
def convert_df_to_csv(df):
    """
    Convertit un DataFrame en CSV pour le téléchargement
    """
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def convert_df_to_excel(df):
    """
    Convertit un DataFrame en Excel pour le téléchargement
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Données')
    return output.getvalue()