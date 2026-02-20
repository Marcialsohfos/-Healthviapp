"""
Module de prédiction pour Streamlit
Version mise à jour
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class VulnerabilityPredictor:
    """Prédicteur de vulnérabilité sanitaire"""
    
    def __init__(self, model_path='ml_model/model_latest.pkl'):
        self.model = None
        self.scaler = None
        self.minmax_scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.metrics = {}
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Charge le modèle et les artefacts"""
        try:
            # Charger le modèle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Charger le préprocessing
            preprocessing_path = model_path.replace('model_', 'preprocessing_')
            if Path(preprocessing_path).exists():
                with open(preprocessing_path, 'rb') as f:
                    preprocessing = pickle.load(f)
                    self.scaler = preprocessing.get('scaler')
                    self.minmax_scaler = preprocessing.get('minmax_scaler')
                    self.label_encoders = preprocessing.get('label_encoders', {})
                    self.feature_names = preprocessing.get('feature_names', [])
                    self.metrics = preprocessing.get('metrics', {})
            
            print(f"✅ Modèle chargé: {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"⚠️  Erreur chargement modèle: {e}")
            self._create_default_model()
    
    def _create_default_model(self):
        """Crée un modèle par défaut"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'larg_voiri_norm', 'mat_mur_score', 'mat_toit_score',
            'eau_bois_score', 'evac_eau_score', 'elec_score',
            'risq_nat_norm', 'score_acces', 'score_habitat',
            'score_services', 'score_risques'
        ]
    
    def preprocess_input(self, input_df):
        """Prétraite les données d'entrée"""
        df = input_df.copy()
        
        # S'assurer que toutes les features sont présentes
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Sélectionner les features dans le bon ordre
        df = df[self.feature_names].fillna(0)
        
        return df
    
    def predict(self, input_df):
        """Fait une prédiction"""
        try:
            # Prétraitement
            processed_data = self.preprocess_input(input_df)
            
            # Normalisation
            if self.scaler is not None:
                processed_scaled = self.scaler.transform(processed_data)
            else:
                processed_scaled = processed_data.values
            
            # Prédiction
            prediction = self.model.predict(processed_scaled)[0]
            
            # S'assurer que la prédiction est dans [0, 1]
            prediction = max(0, min(1, prediction))
            
            # Convertir en pourcentage pour l'affichage
            prediction_pct = prediction * 100
            
            # Déterminer le niveau
            niveau = self._get_level(prediction)
            
            # Calculer la confiance (basée sur la position dans l'arbre)
            confidence = self._calculate_confidence(prediction)
            
            return {
                'success': True,
                'prediction': float(prediction),
                'prediction_pct': float(prediction_pct),
                'niveau': niveau,
                'confidence': confidence,
                'couleur': self._get_color(niveau)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'niveau': 'Inconnu'
            }
    
    def _get_level(self, score):
        """Détermine le niveau de vulnérabilité"""
        if score <= 0.25:
            return "Faible"
        elif score <= 0.50:
            return "Modérée"
        elif score <= 0.75:
            return "Élevée"
        else:
            return "Critique"
    
    def _get_color(self, niveau):
        """Retourne la couleur associée au niveau"""
        colors = {
            'Faible': '#27ae60',    # Vert
            'Modérée': '#f39c12',   # Jaune
            'Élevée': '#e67e22',    # Orange
            'Critique': '#e74c3c',  # Rouge
            'Inconnu': '#95a5a6'    # Gris
        }
        return colors.get(niveau, '#95a5a6')
    
    def _calculate_confidence(self, score):
        """Calcule la confiance de la prédiction"""
        # Plus le score est extrême, plus la confiance est élevée
        if score < 0.2 or score > 0.8:
            return 0.95
        elif score < 0.3 or score > 0.7:
            return 0.85
        else:
            return 0.75
    
    def predict_batch(self, df):
        """Prédit un lot de données"""
        results = []
        
        for idx, row in df.iterrows():
            row_df = pd.DataFrame([row])
            pred = self.predict(row_df)
            
            if pred['success']:
                results.append({
                    'index': idx,
                    'ivs': pred['prediction'],
                    'ivs_pct': pred['prediction_pct'],
                    'niveau': pred['niveau'],
                    'confidence': pred['confidence']
                })
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, top_n=10):
        """Retourne l'importance des features"""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            return importance
        return None

# Instance globale pour Streamlit
_predictor_instance = None

def get_predictor():
    """Retourne l'instance du prédicteur (singleton)"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = VulnerabilityPredictor()
    return _predictor_instance