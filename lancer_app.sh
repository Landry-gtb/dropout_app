#!/bin/bash
# ─────────────────────────────────────────────────────
# Script de lancement — Système d'Alerte Précoce
# ─────────────────────────────────────────────────────

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/dropout_env"
APP_FILE="$APP_DIR/app.py"

echo "═══════════════════════════════════════════════"
echo "  🎓 Système d'Alerte Précoce — Démarrage"
echo "═══════════════════════════════════════════════"

# Vérification de l'environnement virtuel
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR"
fi

# Activation
source "$VENV_DIR/bin/activate"
echo "✅ Environnement virtuel activé"

# Installation des dépendances 
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📥 Installation des dépendances..."
    pip install -r "$APP_DIR/requirements.txt" --quiet
    echo "✅ Dépendances installées"
fi

# Vérification des modèles
echo ""
echo "🔍 Vérification des modèles..."
MISSING=0
for f in \
    "models/RF_Project/Dropout_model" \
    "models/RF_Project/feature_names.pkl" \
    "models/RF_Project/scaler_rf.pkl" \
    "models/RF_Project/seuil_optimal_rf_comport.pkl" \
    "models/DROPOUT_TOOLS/Dropout_model_institutional.pkl" \
    "models/DROPOUT_TOOLS/feature_names_institutional.pkl" \
    "models/DROPOUT_TOOLS/seuil_optimal_fusion.pkl"
do
    if [ -f "$APP_DIR/$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ MANQUANT : $f"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  Certains modèles sont manquants."
    echo "   Copiez-les depuis Google Drive dans le dossier models/"
    echo ""
fi

# Lancement
echo ""
echo "🚀 Lancement de l'application..."
echo "   → Ouvrez http://localhost:8501 dans votre navigateur"
echo "   → Ctrl+C pour arrêter"
echo "═══════════════════════════════════════════════"
echo ""

streamlit run "$APP_FILE" \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
