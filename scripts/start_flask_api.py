#!/usr/bin/env python3
"""Script pour démarrer l'API Flask directement"""

import os
import sys
import signal
import subprocess
import time
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

server = None

def stop_server():
    """Arrête le serveur Flask"""
    global server
    if server:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.terminate()
            print("🛑 Serveur Flask arrêté")
        except:
            pass
        server = None

def start_server():
    """Démarre le serveur Flask directement"""
    global server
    stop_server()
    time.sleep(1.5)
    
    print("🚀 Démarrage du serveur Flask...")
    print("📍 URL: http://127.0.0.1:5000")
    print("🏥 Health check: http://127.0.0.1:5000/health")
    print("🔮 Prédictions: http://127.0.0.1:5000/predict")
    print("📊 Info modèle: http://127.0.0.1:5000/model/info")
    print("📋 Gestion données: http://127.0.0.1:5000/data")
    print("📚 Documentation Swagger: http://127.0.0.1:5000/api/docs")
    print("\\n💡 Utilisez Ctrl+C pour arrêter le serveur\\n")
    
    # Utiliser directement le script run_server.py
    run_server_script = project_root / "src" / "kedro_project" / "api" / "run_server.py"
    
    server = subprocess.Popen(
        f"python {run_server_script}",
        shell=True,
        preexec_fn=os.setsid,
        cwd=project_root
    )
    
    return server

if __name__ == '__main__':
    try:
        server = start_server()
        
        # Attendre que l'utilisateur appuie sur Ctrl+C
        print("Serveur en cours d'exécution. Appuyez sur Ctrl+C pour arrêter...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\n🛑 Arrêt du serveur...")
        stop_server()
        print("✅ Serveur arrêté proprement")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        stop_server()
        sys.exit(1)