#!/usr/bin/env python3
"""Script pour lancer le serveur Flask"""

import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import après modification du PATH
from kedro_project.api.app import app  # noqa: E402

if __name__ == "__main__":
    print("🚀 Démarrage du serveur Flask...")
    print("📍 URL: http://127.0.0.1:5000")
    print("🏥 Health check: http://127.0.0.1:5000/health")
    print("🔮 Prédictions: http://127.0.0.1:5000/predict")
    print("📊 Info modèle: http://127.0.0.1:5000/model/info")
    print("📚 Documentation: http://127.0.0.1:5000/api/docs")
    print("\n💡 Utilisez Ctrl+C pour arrêter le serveur\n")

    app.run(debug=True, host="127.0.0.1", port=5000)
