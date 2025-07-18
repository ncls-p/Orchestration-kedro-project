#!/usr/bin/env python3
"""
Script pour exécuter le pipeline complet Kedro avec logging MLflow
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Exécuter une commande et afficher le résultat"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - Succès")
            if result.stdout:
                print("📋 Sortie:")
                print(result.stdout)
        else:
            print(f"❌ {description} - Échec")
            if result.stderr:
                print("🔴 Erreur:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return False
    
    return True

def main():
    """Fonction principale"""
    print("🎯 Exécution du pipeline complet Kedro avec MLflow")
    print("="*60)
    
    # Liste des pipelines à exécuter dans l'ordre
    pipelines = [
        ("data_processing", "Traitement des données"),
        ("hyperparameter_optimization", "Optimisation des hyperparamètres"),
        ("model_validation", "Validation du modèle"),
        ("model_interpretability", "Interprétabilité du modèle")
    ]
    
    success_count = 0
    total_count = len(pipelines)
    
    # Exécuter chaque pipeline
    for pipeline_name, description in pipelines:
        command = f"uv run kedro run --pipeline={pipeline_name}"
        
        if run_command(command, f"Pipeline: {description}"):
            success_count += 1
            print(f"✅ Pipeline {pipeline_name} terminé avec succès")
        else:
            print(f"❌ Pipeline {pipeline_name} a échoué")
            
        # Pause entre les pipelines
        time.sleep(2)
    
    # Exécuter le pipeline par défaut (tous ensemble)
    print(f"\n{'='*60}")
    print("🎯 Exécution du pipeline par défaut (tous ensemble)")
    print(f"{'='*60}")
    
    default_command = "uv run kedro run"
    if run_command(default_command, "Pipeline par défaut"):
        print("✅ Pipeline par défaut terminé avec succès")
    else:
        print("❌ Pipeline par défaut a échoué")
    
    # Résumé final
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ DE L'EXÉCUTION")
    print(f"{'='*60}")
    print(f"✅ Pipelines réussis: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 Tous les pipelines ont été exécutés avec succès!")
        print("\n📈 Vérifiez les résultats dans MLflow:")
        print("   - Interface MLflow: http://127.0.0.1:5001")
        print("   - Expérience: network-intrusion-detection")
        print("   - Base de données: mlflow.db")
    else:
        print(f"⚠️  {total_count - success_count} pipeline(s) ont échoué")
        print("Vérifiez les logs ci-dessus pour plus de détails")
    
    # Informations supplémentaires
    print(f"\n{'='*60}")
    print("🔗 LIENS UTILES")
    print(f"{'='*60}")
    print("📊 MLflow UI: http://127.0.0.1:5001")
    print("🌐 API Flask: http://127.0.0.1:5000 (si démarrée)")
    print("📚 Documentation API: http://127.0.0.1:5000/api/docs")
    print("🎯 Kedro Viz: kedro viz")

if __name__ == "__main__":
    main()