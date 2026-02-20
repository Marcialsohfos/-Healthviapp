"""
Script d'entraînement du modèle ML pour Streamlit
Version mise à jour avec les résultats de l'étude
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

class VulnerabilityModel:
    """Modèle de vulnérabilité basé sur l'étude 2026"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.metrics = {}
        self.feature_names = []
        self.cluster_info = {}
        
    def load_data(self, filepath):
        """Charge les données Excel"""
        print(f"📂 Chargement de {filepath}")
        
        try:
            df = pd.read_excel(filepath)
            print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Nettoyage des doublons si nécessaire
            df = df.drop_duplicates(subset='id_poche', keep='first')
            print(f"   → {df.shape[0]} poches uniques après dédoublonnage")
            
            return df
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def calculate_scores(self, df):
        """Calcule les scores composites basés sur l'étude"""
        df_scored = df.copy()
        
        # 1. Score d'accès
        access_vars = []
        
        if 'larg_voiri' in df_scored.columns:
            df_scored['larg_voiri'] = pd.to_numeric(df_scored['larg_voiri'], errors='coerce').fillna(0)
            df_scored['larg_voiri_norm'] = self.minmax_scaler.fit_transform(df_scored[['larg_voiri']]).flatten()
            access_vars.append('larg_voiri_norm')
        
        for dist in ['dist_sant', 'dist_sec', 'dist_ecole']:
            if dist in df_scored.columns:
                df_scored[dist] = pd.to_numeric(df_scored[dist], errors='coerce').fillna(df_scored[dist].median())
                df_scored[f'{dist}_inv'] = 1 / (df_scored[dist] + 1)
                df_scored[f'{dist}_norm'] = self.minmax_scaler.fit_transform(df_scored[[f'{dist}_inv']]).flatten()
                access_vars.append(f'{dist}_norm')
        
        if access_vars:
            df_scored['score_acces'] = df_scored[access_vars].mean(axis=1)
        else:
            df_scored['score_acces'] = 0.5
        
        # 2. Score habitat
        habitat_vars = []
        
        if 'mat_mur' in df_scored.columns:
            mur_map = {
                'Terre': 0.2, 'Banco': 0.2, 'Planche': 0.3,
                'Brique terre': 0.5, 'Parpaing': 0.8, 'Béton': 1.0
            }
            df_scored['mat_mur_score'] = df_scored['mat_mur'].astype(str).map(mur_map).fillna(0.3)
            habitat_vars.append('mat_mur_score')
        
        if 'mat_toit' in df_scored.columns:
            toit_map = {
                'Chaume': 0.2, 'Tôle': 0.6, 'Bac alu': 0.7,
                'Tuile': 0.9, 'Béton': 1.0
            }
            df_scored['mat_toit_score'] = df_scored['mat_toit'].astype(str).map(toit_map).fillna(0.5)
            habitat_vars.append('mat_toit_score')
        
        if habitat_vars:
            df_scored['score_habitat'] = df_scored[habitat_vars].mean(axis=1)
        else:
            df_scored['score_habitat'] = 0.5
        
        # 3. Score services
        services_vars = []
        
        if 'eau_bois' in df_scored.columns:
            eau_map = {'Oui': 1.0, 'Non': 0.0, 'Partiel': 0.5}
            df_scored['eau_bois_score'] = df_scored['eau_bois'].astype(str).map(eau_map).fillna(0)
            services_vars.append('eau_bois_score')
        
        if 'evac_eau' in df_scored.columns:
            evac_map = {'Réseau': 1.0, 'Fossé': 0.5, 'Nature': 0.2, 'Aucune': 0.0}
            df_scored['evac_eau_score'] = df_scored['evac_eau'].astype(str).map(evac_map).fillna(0.3)
            services_vars.append('evac_eau_score')
        
        if 'elec' in df_scored.columns:
            elec_map = {'Oui': 1.0, 'Non': 0.0, 'Partiel': 0.5}
            df_scored['elec_score'] = df_scored['elec'].astype(str).map(elec_map).fillna(0)
            services_vars.append('elec_score')
        
        if services_vars:
            df_scored['score_services'] = df_scored[services_vars].mean(axis=1)
        else:
            df_scored['score_services'] = 0.5
        
        # 4. Score risques
        risques_vars = []
        
        if 'risq_nat' in df_scored.columns:
            risque_map = {
                'Inondation': 0.8, 'Glissement': 0.7, 'Éboulement': 0.6,
                'Inondation/Glissement': 0.9, 'Aucun': 0.1
            }
            df_scored['risq_nat_score'] = df_scored['risq_nat'].astype(str).map(risque_map).fillna(0.5)
            df_scored['risq_nat_norm'] = self.minmax_scaler.fit_transform(df_scored[['risq_nat_score']]).flatten()
            risques_vars.append('risq_nat_norm')
        
        if risques_vars:
            df_scored['score_risques'] = df_scored[risques_vars].mean(axis=1)
        else:
            df_scored['score_risques'] = 0.5
        
        # IVS (Indice de Vulnérabilité Sanitaire)
        df_scored['ivs_brut'] = df_scored[['score_acces', 'score_habitat', 'score_services', 'score_risques']].mean(axis=1)
        df_scored['ivs_normalise'] = self.minmax_scaler.fit_transform(df_scored[['ivs_brut']]).flatten()
        
        # Niveaux de vulnérabilité (basés sur les quartiles de l'étude)
        df_scored['niveau_vulnerabilite'] = pd.cut(
            df_scored['ivs_normalise'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Faible', 'Modérée', 'Élevée', 'Critique']
        )
        
        return df_scored
    
    def assign_clusters(self, df):
        """Assigne les clusters basés sur la classification de l'étude"""
        df_clustered = df.copy()
        
        # Créer 4 clusters basés sur l'IVS (comme dans l'étude)
        try:
            df_clustered['Cluster'] = pd.qcut(
                df_clustered['ivs_normalise'], 
                q=4, 
                labels=[4, 3, 2, 1],  # 1 = plus vulnérable
                duplicates='drop'
            ).astype(int)
        except:
            # Fallback
            percentiles = df_clustered['ivs_normalise'].quantile([0, 0.25, 0.5, 0.75, 1]).values
            percentiles = np.unique(percentiles)
            df_clustered['Cluster'] = pd.cut(
                df_clustered['ivs_normalise'],
                bins=percentiles,
                labels=[4, 3, 2, 1],
                include_lowest=True
            ).astype(int)
        
        # Libellés des clusters
        cluster_labels = {
            1: "Très élevé - URGENT",
            2: "Élevé - HAUTE",
            3: "Modéré - MOYENNE",
            4: "Faible - FAIBLE"
        }
        df_clustered['cluster_libelle'] = df_clustered['Cluster'].map(cluster_labels)
        
        # Couleurs associées
        cluster_colors = {
            1: "#e74c3c",  # Rouge
            2: "#e67e22",  # Orange
            3: "#f39c12",  # Jaune
            4: "#27ae60"   # Vert
        }
        df_clustered['cluster_couleur'] = df_clustered['Cluster'].map(cluster_colors)
        
        return df_clustered
    
    def prepare_features(self, df):
        """Prépare les features pour l'entraînement"""
        # Features numériques
        numeric_features = [
            'larg_voiri_norm', 'mat_mur_score', 'mat_toit_score',
            'eau_bois_score', 'evac_eau_score', 'elec_score',
            'risq_nat_norm', 'score_acces', 'score_habitat',
            'score_services', 'score_risques'
        ]
        
        # Garder uniquement les features existantes
        self.feature_names = [f for f in numeric_features if f in df.columns]
        
        X = df[self.feature_names].fillna(0)
        y = df['ivs_normalise'].fillna(0)
        
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Échantillons: {len(X)}")
        
        return X, y
    
    def train_model(self, X, y):
        """Entraîne le modèle avec les meilleurs paramètres de l'étude"""
        print("🎯 Entraînement du modèle...")
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest avec les paramètres optimaux de l'étude
        rf_model = RandomForestRegressor(
            n_estimators=200,
            min_samples_split=2,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost avec les paramètres optimaux
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Entraîner les deux modèles
        rf_model.fit(X_train_scaled, y_train)
        xgb_model.fit(X_train_scaled, y_train)
        
        # Évaluer
        rf_pred = rf_model.predict(X_test_scaled)
        xgb_pred = xgb_model.predict(X_test_scaled)
        
        rf_r2 = r2_score(y_test, rf_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        # Choisir le meilleur
        if rf_r2 >= xgb_r2:
            self.model = rf_model
            best_model_name = "Random Forest"
            self.metrics = {
                'r2': rf_r2,
                'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'mae': mean_absolute_error(y_test, rf_pred),
                'best_model': best_model_name
            }
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            self.model = xgb_model
            best_model_name = "XGBoost"
            self.metrics = {
                'r2': xgb_r2,
                'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'mae': mean_absolute_error(y_test, xgb_pred),
                'best_model': best_model_name
            }
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.metrics.update({
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
        
        print(f"✅ Modèle entraîné: {best_model_name} (R²: {self.metrics['r2']:.3f})")
        
        return self.model
    
    def save_model(self, output_dir='ml_model'):
        """Sauvegarde le modèle et tous les artefacts"""
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sauvegarde du modèle
        model_path = f'{output_dir}/model_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Sauvegarde des objets de préprocessing
        preprocessing_path = f'{output_dir}/preprocessing_{timestamp}.pkl'
        with open(preprocessing_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'minmax_scaler': self.minmax_scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance
            }, f)
        
        # Sauvegarde des métriques
        metrics_path = f'{output_dir}/metrics_{timestamp}.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        # Version latest
        latest_model = f'{output_dir}/model_latest.pkl'
        latest_preprocessing = f'{output_dir}/preprocessing_latest.pkl'
        
        import shutil
        shutil.copy2(model_path, latest_model)
        shutil.copy2(preprocessing_path, latest_preprocessing)
        
        print(f"💾 Modèle sauvegardé dans {output_dir}/")
        return model_path

def train_model():
    """Fonction principale pour l'entraînement"""
    print("="*60)
    print("ENTRAÎNEMENT DU MODÈLE DE VULNÉRABILITÉ SANITAIRE")
    print("="*60)
    
    # Initialiser le modèle
    model = VulnerabilityModel()
    
    # Chercher le fichier de données
    data_paths = [
        'data/bdpoche_prec_corrige_20260218_171512.xlsx',
        'data/bdpoche_prec.xlsx',
        'bdpoche_prec_corrige_20260218_171512.xlsx',
        'bdpoche_prec.xlsx'
    ]
    
    df = None
    for path in data_paths:
        if Path(path).exists():
            df = model.load_data(path)
            if df is not None:
                break
    
    if df is None:
        print("❌ Aucun fichier de données trouvé")
        return None
    
    # Calculer les scores
    df_scored = model.calculate_scores(df)
    
    # Assigner les clusters
    df_clustered = model.assign_clusters(df_scored)
    
    # Préparer les features
    X, y = model.prepare_features(df_clustered)
    
    # Entraîner le modèle
    model.train_model(X, y)
    
    # Sauvegarder
    model.save_model()
    
    # Afficher les résultats
    print("\n📊 RÉSULTATS:")
    print(f"  - Modèle: {model.metrics.get('best_model', 'Inconnu')}")
    print(f"  - R²: {model.metrics.get('r2', 0):.3f}")
    print(f"  - RMSE: {model.metrics.get('rmse', 0):.2f}")
    print(f"  - MAE: {model.metrics.get('mae', 0):.2f}")
    print(f"  - Features: {model.metrics.get('n_features', 0)}")
    print(f"  - Échantillons: {model.metrics.get('n_samples', 0)}")
    
    if model.feature_importance is not None:
        print("\n🔝 TOP 5 FACTEURS DE VULNÉRABILITÉ:")
        for idx, row in model.feature_importance.head().iterrows():
            print(f"  - {row['feature']}: {row['importance']:.3f}")
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    
    return model

if __name__ == "__main__":
    train_model()