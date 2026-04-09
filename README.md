# 🎓 Système de Prédiction Précoce du Décrochage Académique

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-interpretability-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Fusion pondérée de deux modèles Random Forest — Données comportementales + Données institutionnelles**

*Master Systèmes Intelligents et Multimédia — Vietnam National University, Hanoi (VNU)*

</div>

---

## 📋 Table des matières

- [Présentation du projet](#-présentation-du-projet)
- [Architecture du système](#-architecture-du-système)
- [Datasets](#-datasets)
- [Pipeline ML](#-pipeline-ml)
- [Résultats](#-résultats)
- [Application Streamlit](#-application-streamlit)
- [Structure du dépôt](#-structure-du-dépôt)
- [Installation et démarrage](#-installation-et-démarrage)
- [Fichiers modèles requis](#-fichiers-modèles-requis)
- [Utilisation](#-utilisation)
- [Technologies](#-technologies)
- [Limitations connues](#-limitations-connues)
- [Auteur](#-auteur)

---

## 🎯 Présentation du projet

Ce projet développe un **système de prédiction précoce du décrochage académique** destiné à aider le personnel pédagogique à identifier les étudiants à risque et à intervenir de manière ciblée.

### Approche

Le système repose sur la **fusion pondérée de deux modèles Random Forest indépendants** entraînés sur deux sources de données complémentaires :

| Source | Description | Taille |
|---|---|---|
| **Comportementale** | Questionnaire structuré collecté auprès d'étudiants (données primaires) | 564 observations |
| **Institutionnelle** | Dataset UCI — *Predict Students' Dropout and Academic Success* (Realinho et al., 2022) | 4 424 observations |

### Formule de fusion

```
Risque(x) = 0.60 × p_RF_comportemental(x) + 0.40 × p_RF_institutionnel(x)
```

- **Seuil de décision optimal** : θ = 0.6075 (calculé sur le jeu de validation)
- **Décision** : Risque ≥ 0.6075 → 🔴 ALERTE | Risque < 0.6075 → 🟢 SÉCURITÉ

---

## 🏗 Architecture du système

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTÈME D'ALERTE PRÉCOCE                     │
├──────────────────────────┬──────────────────────────────────────┤
│   PARTIE 1               │   PARTIE 2                          │
│   Dataset Comportemental │   Dataset Institutionnel (UCI)       │
│   564 obs · 26 features  │   4 424 obs · 34 features           │
│         ↓                │         ↓                           │
│   Encodage Ordinal       │   One-Hot Encoding                  │
│   + One-Hot Encoding     │   (276 features finales)            │
│         ↓                │         ↓                           │
│   Split 70/15/15         │   Split 70/15/15                    │
│   (stratifié)            │   (stratifié)                       │
│         ↓                │         ↓                           │
│   StandardScaler         │   [pas de scaling]                  │
│   (fit sur X_train)      │                                     │
│         ↓                │         ↓                           │
│   SMOTE (X_train only)   │   SMOTE (X_train only)              │
│         ↓                │         ↓                           │
│   RF Comportemental      │   RF Institutionnel                 │
│   RandomizedSearchCV     │   RandomizedSearchCV                │
│   OOB = 0.99             │   OOB = 0.91                        │
│         ↓                │         ↓                           │
│   Score p_RF             │   Score p_inst                      │
│   Seuil = 0.7843         │   Seuil = 0.4307                   │
│         ↓                │         ↓                           │
├──────────────────────────┴──────────────────────────────────────┤
│              FUSION PONDÉRÉE (60% / 40%)                        │
│         θ_fusion = (0.7843 + 0.4307) / 2 = 0.6075             │
│                          ↓                                      │
│              🔴 ALERTE  /  🟢 SÉCURITÉ                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Datasets

### Dataset comportemental (collecte primaire)

Constitué via un questionnaire structuré administré en ligne auprès d'étudiants d'enseignement supérieur.

| Dimension | Variables |
|---|---|
| Profil général | Tranche d'âge, genre, niveau d'étude, type/mention Bac |
| Comportement académique | Moyenne, matières échouées, absences, heures d'étude, participation, devoirs, procrastination |
| État psychologique | Motivation, stress, confiance en soi, fatigue mentale |
| Contexte socio-économique | Statut boursier, activité salariée, niveau d'étude des parents |
| Contexte de vie | Internet, environnement de travail, soutien entourage, logement, trajet, LMS |
| **Cible** | Abandon envisagé : **Oui (33%) / Non (67%)** |

### Dataset institutionnel (UCI)

> Realinho, V., Vieira Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting student dropout and academic success.* Data, 7(11), 146.

Variable cible recodée en binaire : **Dropout = 1** | Graduate/Enrolled = 0

Distribution : 1 421 Dropout (32%) / 3 003 Non-Dropout (68%)

---

## ⚙️ Pipeline ML

### Prétraitement

```python
# Comportemental — Étapes dans l'ordre
1. Normalisation apostrophes typographiques (U+2019 → U+0027)
2. Encodage ordinal (20 variables avec ordre naturel)
3. One-Hot Encoding (6 variables nominales, drop_first=True)
4. Split stratifié 70 / 15 / 15
5. StandardScaler fitté sur X_train uniquement  ← anti-data leakage
6. SMOTE sur X_train post-split et post-scaling

# Institutionnel
1. Binarisation de la cible (Dropout=1, autres=0)
2. Conversion catégorielles numériques → object
3. One-Hot Encoding (276 features finales)
4. Split stratifié 70 / 15 / 15
5. SMOTE sur X_train (pas de scaling pour RF)
```

### Optimisation des hyperparamètres

`RandomizedSearchCV` — n_iter=80 — StratifiedKFold(5) — scoring=F1

| Paramètre | RF Comportemental | RF Institutionnel |
|---|---|---|
| `n_estimators` | 149 | [100, 600] |
| `max_depth` | 7 | [None, 10, 20, 30, 40] |
| `min_samples_leaf` | 3 | [1, 6] |
| `max_features` | 0.3 | ['sqrt', 'log2', 0.3, 0.5] |
| `max_samples` | 0.8 | — |

### Calcul des seuils optimaux

Seuils calculés par maximisation du F1 sur la **courbe Précision-Rappel appliquée au jeu de validation** — jamais sur le jeu de test.

```python
prec, rec, thresh = precision_recall_curve(y_val, y_proba_val)
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
seuil_optimal = thresh[np.argmax(f1[:-1])]
```

### Validation croisée (RF Comportemental)

Validation croisée stratifiée à **10 folds** avec SMOTE appliqué **à l'intérieur de chaque fold** via `imblearn.Pipeline` — garantit l'absence de data leakage.

---

## 📈 Résultats

### RF Comportemental — Validation croisée 10 folds

| Métrique | Valeur | Écart-type |
|---|---|---|
| Accuracy | 0.9929 | ±0.0118 |
| F1 | 0.9895 | ±0.0175 |
| Précision | 0.9797 | ±0.0334 |
| Rappel | 1.0000 | ±0.0000 |
| AUC-ROC | 1.0000 | ±0.0000 |
| OOB Score | 0.9905 | — |

> **Note :** Les performances élevées reflètent la forte valeur prédictive des features comportementales (corrélation max = 0.74 avec la cible). Le gap F1 Train/Val = 0.032 confirme l'absence de surapprentissage. À interpréter avec prudence en raison de la taille limitée du jeu de test (85 observations).

### RF Institutionnel — Jeu de test (664 observations)

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| Non-Dropout | 0.90 | 0.92 | 0.91 | 451 |
| Dropout | 0.83 | 0.78 | 0.81 | 213 |
| **Global** | **0.88** | **0.88** | **0.88** | **664** |

- **Accuracy** : 87.95%
- **AUC-ROC** : 0.9339
- **OOB Score** : 0.9134
- **Seuil optimal** : 0.4307

### Seuils de décision

| Modèle | Seuil optimal | Fichier sauvegardé |
|---|---|---|
| RF Comportemental | 0.7843 | `seuil_optimal_rf_comport.pkl` |
| RF Institutionnel | 0.4307 | `seuil_optimal_rf_institutional.pkl` |
| **Fusion** | **0.6075** | **`seuil_optimal_fusion.pkl`** |

---

## 🖥 Application Streamlit

Interface web interactive déployée localement sur `localhost:8501`.

### Fonctionnalités

| Mode | Description |
|---|---|
| 📝 **Saisie manuelle** | Formulaire 5 onglets — 26 champs comportementaux + 13 institutionnels |
| 📁 **Import batch** | CSV/XLSX — template avec listes déroulantes — export CSV résultats |
| 🔍 **SHAP interactif** | 2 graphiques côte à côte (RF comport. + RF inst.) — top 8 features |
| 💡 **Recommandations** | 4 recommandations pédagogiques dynamiques basées sur les SHAP values |
| 📄 **Export PDF** | Fiche individuelle : statut, scores, profil étudiant, recommandations |
| 🕐 **Historique** | Toutes les analyses de la session avec consultation |

### Aperçu de l'interface

<img width="1600" height="783" alt="image" src="https://github.com/user-attachments/assets/2386f7f7-507c-4e90-8423-7032c54993c1" />


---

## 📁 Structure du dépôt

```
dropout_app/
│
├── app.py                          # Application Streamlit principale
├── lancer_app.sh                   # Script de démarrage WSL2/Linux
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
│
├── models/
│   ├── RF_Project/
│   │   ├── Dropout_model           # RF comportemental entraîné
│   │   ├── scaler_rf.pkl           # StandardScaler (31 features)
│   │   ├── feature_names.pkl       # Liste des 31 features comportementales
│   │   └── seuil_optimal_rf_comport.pkl   # Seuil optimal = 0.7843
│   │
│   └── DROPOUT_TOOLS/
│       ├── Dropout_model_institutional.pkl  # RF institutionnel entraîné
│       ├── feature_names_institutional.pkl  # 276 features institutionnelles
│       └── seuil_optimal_fusion.pkl         # Seuil de fusion = 0.6075
│
└── notebooks/
    ├── Dropout_Pipeline_Final.ipynb    # Pipeline complet (Colab)
    └── Partie2_RF_Institutionnel.ipynb # Pipeline RF institutionnel
```

> ⚠️ **Les fichiers `.pkl` ne sont pas inclus dans le dépôt** (taille). Voir la section [Fichiers modèles requis](#-fichiers-modèles-requis).

---

## 🚀 Installation et démarrage

### Prérequis

- Python 3.12
- WSL2 (Ubuntu) recommandé sur Windows, ou Linux natif
- Google Colab (pour l'entraînement des modèles)

### 1. Cloner le dépôt

```bash
git clone https://github.com/Landry-gtb/dropout_app.git
cd dropout_app
```

### 2. Créer l'environnement virtuel

```bash
python3 -m venv ~/dropout_env
source ~/dropout_env/bin/activate
pip install -r requirements.txt
```

### 3. Placer les fichiers modèles

Télécharger les fichiers `.pkl` depuis Google Drive et les placer dans les dossiers correspondants (voir structure ci-dessus).

### 4. Lancer l'application

```bash
# Méthode recommandée (script de lancement)
./lancer_app.sh

# Ou directement
source ~/dropout_env/bin/activate
streamlit run app.py --server.port 8501
```

Ouvrir `http://localhost:8501` dans le navigateur.

---

## 📦 Fichiers modèles requis

Les modèles sont entraînés sur **Google Colab** et doivent être copiés localement :

| Fichier | Répertoire Google Drive | Répertoire local |
|---|---|---|
| `Dropout_model` | `MyDrive/RF_Project/` | `models/RF_Project/` |
| `scaler_rf.pkl` | `MyDrive/RF_Project/` | `models/RF_Project/` |
| `feature_names.pkl` | `MyDrive/RF_Project/` | `models/RF_Project/` |
| `seuil_optimal_rf_comport.pkl` | `MyDrive/RF_Project/` | `models/RF_Project/` |
| `Dropout_model_institutional.pkl` | `MyDrive/DROPOUT_TOOLS/` | `models/DROPOUT_TOOLS/` |
| `feature_names_institutional.pkl` | `MyDrive/DROPOUT_TOOLS/` | `models/DROPOUT_TOOLS/` |
| `seuil_optimal_fusion.pkl` | `MyDrive/DROPOUT_TOOLS/` | `models/DROPOUT_TOOLS/` |

> ⚠️ **Version sklearn** : les modèles ont été entraînés avec `scikit-learn==1.6.1`. Un avertissement de version apparaît si l'environnement local utilise une version différente — le fonctionnement reste opérationnel.

---

## 💻 Utilisation

### Saisie manuelle d'un étudiant

1. Choisir le mode **Saisie manuelle**
2. Remplir les 5 onglets : Profil général → Parcours académique → Profil psychologique → Contexte de vie → Données institutionnelles
3. Cliquer sur **Analyser le risque de décrochage**
4. Consulter le score, les graphiques SHAP et les recommandations
5. Télécharger la fiche PDF si nécessaire

### Import batch (plusieurs étudiants)

1. Choisir le mode **Import fichier**
2. Télécharger le **template Excel** (listes déroulantes incluses)
3. Remplir le template et l'importer
4. Consulter le tableau de résultats trié par risque décroissant
5. Exporter les résultats en CSV

### Entraînement des modèles (Google Colab)

```
1. Ouvrir Dropout_Pipeline.ipynb sur Google Colab
2. Monter Google Drive
3. Placer les datasets dans MyDrive/DROPOUT_TOOLS/
4. Exécuter les cellules dans l'ordre :
   - Partie 1 : RF Comportemental
   - Partie 2 : RF Institutionnel
   - Partie 3 : Système d'alerte + calcul seuil de fusion
5. Copier les .pkl générés vers l'environnement local
```

---

## 🛠 Technologies

| Catégorie | Bibliothèque | Version |
|---|---|---|
| ML | scikit-learn | 1.6.1 |
| Rééchantillonnage | imbalanced-learn | latest |
| Interprétabilité | shap | latest |
| Interface | streamlit | latest |
| Données | pandas, numpy | latest |
| Visualisation | matplotlib, seaborn | latest |
| PDF | fpdf2 | latest |
| Excel | openpyxl | latest |
| Sérialisation | joblib | latest |

---

## ⚠️ Limitations connues

1. **Taille du dataset comportemental** : 564 observations — jeu de test de 85 lignes seulement. Les métriques à 100% doivent être interprétées avec prudence.

2. **Populations hétérogènes** : les deux datasets proviennent de contextes différents (étudiants locaux vs institution portugaise). Le seuil de fusion est une approximation (moyenne des seuils individuels) car une optimisation directe sur un score fusionné commun est impossible.

3. **Calibration des probabilités** : le RF comportemental produit des probabilités très polarisées (proches de 0 ou 1), ce qui peut déséquilibrer le score fusionné malgré les poids.

4. **Biais de désirabilité sociale** : les réponses au questionnaire comportemental peuvent être biaisées par le contexte d'évaluation.

5. **Version sklearn** : les modèles `.pkl` sont liés à scikit-learn 1.6.1 — une incompatibilité de version peut apparaître sur des environnements plus récents.

---

## 📚 Référence principale

```bibtex
@article{realinho2022,
  author  = {Realinho, Valentim and Vieira Machado, Jorge and Baptista, Luís and Martins, Mónica V.},
  title   = {Predicting Student Dropout and Academic Success},
  journal = {Data},
  volume  = {7},
  number  = {11},
  pages   = {146},
  year    = {2022},
  doi     = {10.3390/data7110146}
}
```

---

## 👤 Auteur

**IRUMVA Landry (Landros)**
- GitHub : [@Landry-gtb](https://github.com/Landry-gtb)
- Programme : Master Systèmes Intelligents et Multimédia (SIM)
- Institution : Vietnam National University — Hanoi (VNU)

---

<div align="center">

*Ce système est un outil d'aide à la décision. Il ne remplace pas le jugement pédagogique humain.*

</div>
