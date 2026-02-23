"""
Constantes de l'application - Version mise à jour
"""

# Seuils de vulnérabilité (basés sur l'IVS normalisé)
VULNERABILITY_THRESHOLDS = {
    'FAIBLE': 0.25,
    'MODEREE': 0.50,
    'ELEVEE': 0.75,
    'CRITIQUE': 1.00
}

VULNERABILITY_LABELS = {
    'FAIBLE': 'Faible',
    'MODEREE': 'Modérée',
    'ELEVEE': 'Élevée',
    'CRITIQUE': 'Critique'
}

# Couleurs associées aux niveaux
VULNERABILITY_COLORS = {
    'Faible': '#27ae60',    # Vert
    'Modérée': '#f39c12',   # Jaune
    'Élevée': '#e67e22',    # Orange
    'Critique': '#e74c3c',  # Rouge
    'Inconnu': '#95a5a6'    # Gris
}

# Couleurs des clusters
CLUSTER_COLORS = {
    1: '#e74c3c',  # Rouge - Très élevé
    2: '#e67e22',  # Orange - Élevé
    3: '#f39c12',  # Jaune - Modéré
    4: '#27ae60'   # Vert - Faible
}

CLUSTER_LABELS = {
    1: "Très élevé - URGENT",
    2: "Élevé - HAUTE", 
    3: "Modéré - MOYENNE",
    4: "Faible - FAIBLE"
}

# Villes
VILLES = ['Douala', 'Yaoundé']

# Communes de Yaoundé
COMMUNES_YAOUNDE = [
    "Yaoundé I", "Yaoundé II", "Yaoundé III", "Yaoundé IV",
    "Yaoundé V", "Yaoundé VI", "Yaoundé VII"
]

# Communes de Douala
COMMUNES_DOUALA = [
    "Douala I", "Douala II", "Douala III", "Douala IV", "Douala V"
]

# Top poches les plus vulnérables (résultats de l'étude)
TOP_POCHES_VULNERABLES = {
    'Douala': [
        {'nom': 'bonendale b', 'ivs': 1.000, 'risque': 'Inondation'},
        {'nom': 'minkwelle', 'ivs': 1.000, 'risque': 'Inondation'},
        {'nom': 'bonendale a', 'ivs': 1.000, 'risque': 'Inondation'},
        {'nom': 'mabanda b', 'ivs': 0.667, 'risque': 'Inondation/Éboulement'},
        {'nom': 'mabanda c', 'ivs': 0.667, 'risque': 'Inondation/Éboulement'}
    ],
    'Yaoundé': [
        {'nom': 'Ahala b', 'ivs': 0.584, 'risque': 'Inondation'},
        {'nom': 'Emana-bilono', 'ivs': 0.584, 'risque': 'Inondation'},
        {'nom': 'Obili b', 'ivs': 0.584, 'risque': 'Inondation'},
        {'nom': 'Ekounou Deux chevaux', 'ivs': 0.584, 'risque': 'Inondation'},
        {'nom': 'Etoa a', 'ivs': 0.584, 'risque': 'Inondation'}
    ]
}

# Facteurs clés par type de risque
FACTEURS_CLES = {
    'Inondation': [
        "Exposition aux inondations",
        "Absence de drainage",
        "Habitat en zone inondable"
    ],
    'Glissement': [
        "Pente forte",
        "Instabilité des sols",
        "Absence de retenue"
    ],
    'Éboulement': [
        "Risque d'éboulement",
        "Proximité de falaises",
        "Végétation insuffisante"
    ],
    'Défaut_accès': [
        "Enclavement",
        "Voirie dégradée",
        "Distance aux services"
    ]
}

# Recommandations par niveau
RECOMMENDATIONS = {
    'Faible': [
        "✅ Surveillance préventive",
        "✅ Entretien régulier des infrastructures",
        "✅ Sensibilisation communautaire"
    ],
    'Modérée': [
        "🟡 Amélioration des infrastructures de base",
        "🟡 Plan de gestion des risques",
        "🟡 Renforcement des capacités locales"
    ],
    'Élevée': [
        "🟠 Interventions prioritaires",
        "🟠 Systèmes d'alerte précoce",
        "🟠 Relocalisation partielle si nécessaire"
    ],
    'Critique': [
        "🔴 INTERVENTION URGENTE REQUISE",
        "🔴 Évacuation préventive recommandée",
        "🔴 Plan d'urgence immédiat",
        "🔴 Reconstruction des infrastructures"
    ]
}

# Variables essentielles pour le modèle
ESSENTIAL_FEATURES = [
    'larg_voiri', 'mat_mur', 'mat_toit',
    'eau_bois', 'evac_eau', 'elec',
    'risq_nat', 'dist_sant', 'dist_ecole'
]

# Configuration de l'application
APP_CONFIG = {
    'NAME': 'IA Vulnérabilité Sanitaire - Cameroun',
    'VERSION': '2.0.0',
    'AUTHOR': 'Équipe de Recherche - Lab_Math and CIE-SCSM Group',
    'YEAR': '2026',
    'DATA_SOURCE': 'Enquête 2025 - 266 poches d\'habitat précaire',
    'MAX_FILE_SIZE_MB': 100,
    'SUPPORTED_FORMATS': ['.xlsx', '.xls', '.csv']
}

# Messages d'erreur
ERROR_MESSAGES = {
    'NO_DATA': "❌ Aucune donnée disponible. Veuillez charger les données d'abord.",
    'NO_MODEL': "❌ Modèle non chargé. Veuillez entraîner ou charger un modèle.",
    'FILE_NOT_FOUND': "❌ Fichier non trouvé: {filename}",
    'INVALID_FORMAT': "❌ Format de fichier non supporté.",
    'UPLOAD_ERROR': "❌ Erreur lors du téléversement du fichier.",
    'PREDICTION_ERROR': "❌ Erreur lors de la prédiction: {error}"
}