#!/usr/bin/env python3
"""
Exemples d'utilisation de l'API Flask pour la détection d'intrusion réseau
"""

import json
import requests
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://127.0.0.1:5000"
HEADERS = {"Content-Type": "application/json"}


def test_health_check():
    """Test du health check de l'API"""
    print("🏥 Test du health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_model_info():
    """Test des informations du modèle"""
    print("📊 Test des informations du modèle...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_single_prediction():
    """Test de prédiction individuelle"""
    print("🔮 Test de prédiction individuelle...")
    
    # Exemple de trafic suspect
    suspicious_data = {
        "timestamp": "2024-01-15T14:30:00Z",
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.1",
        "source_port": 12345,
        "destination_port": 22,  # SSH - suspect
        "protocol": "TCP",
        "payload_size": 0,  # Payload vide - suspect
        "user_agent": "bot-scanner",  # User agent suspect
        "status": "failed"  # Statut d'échec - suspect
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers=HEADERS,
        json=suspicious_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_batch_prediction():
    """Test de prédictions par lot"""
    print("📦 Test de prédictions par lot...")
    
    batch_data = {
        "data": [
            {
                "timestamp": "2024-01-15T14:30:00Z",
                "source_ip": "192.168.1.100",
                "destination_ip": "10.0.0.1",
                "source_port": 12345,
                "destination_port": 80,  # HTTP - normal
                "protocol": "TCP",
                "payload_size": 1024,
                "user_agent": "Mozilla/5.0",
                "status": "success"
            },
            {
                "timestamp": "2024-01-15T14:31:00Z",
                "source_ip": "192.168.1.101",
                "destination_ip": "10.0.0.2",
                "source_port": 54321,
                "destination_port": 445,  # SMB - suspect
                "protocol": "TCP",
                "payload_size": 15000,  # Payload large - suspect
                "user_agent": "exploit-scanner",  # User agent suspect
                "status": "failed"
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        headers=HEADERS,
        json=batch_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_data_management():
    """Test des opérations CRUD sur les données"""
    print("📋 Test de gestion des données...")
    
    # Données d'exemple
    sample_data = {
        "timestamp": "2024-01-15T15:00:00Z",
        "source_ip": "192.168.1.200",
        "destination_ip": "10.0.0.5",
        "source_port": 8080,
        "destination_port": 443,
        "protocol": "TCP",
        "payload_size": 512,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "status": "success"
    }
    
    # 1. Lister les données (vides au début)
    print("📄 Listing des données...")
    response = requests.get(f"{BASE_URL}/data")
    print(f"Status: {response.status_code}")
    print(f"Données existantes: {len(response.json())}")
    
    # 2. Ajouter des données
    print("➕ Ajout de données...")
    response = requests.post(
        f"{BASE_URL}/data",
        headers=HEADERS,
        json=sample_data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # 3. Modifier des données
    print("✏️ Modification de données...")
    update_data = {
        "timestamp": sample_data["timestamp"],
        "source_ip": sample_data["source_ip"],
        "destination_ip": sample_data["destination_ip"],
        "payload_size": 1024,  # Nouvelle taille
        "status": "modified"
    }
    
    response = requests.patch(
        f"{BASE_URL}/data",
        headers=HEADERS,
        json=update_data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # 4. Lister les données modifiées
    print("📄 Listing des données modifiées...")
    response = requests.get(f"{BASE_URL}/data")
    print(f"Status: {response.status_code}")
    print(f"Données: {json.dumps(response.json(), indent=2)}")
    
    # 5. Supprimer des données
    print("🗑️ Suppression de données...")
    delete_data = {
        "timestamp": sample_data["timestamp"],
        "source_ip": sample_data["source_ip"],
        "destination_ip": sample_data["destination_ip"]
    }
    
    response = requests.delete(
        f"{BASE_URL}/data",
        headers=HEADERS,
        json=delete_data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_error_handling():
    """Test de gestion des erreurs"""
    print("❌ Test de gestion des erreurs...")
    
    # Test avec données incomplètes
    print("🔍 Test avec données incomplètes...")
    incomplete_data = {
        "timestamp": "2024-01-15T14:30:00Z",
        "source_ip": "192.168.1.100"
        # Champs manquants
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers=HEADERS,
        json=incomplete_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test avec types invalides
    print("🔍 Test avec types invalides...")
    invalid_data = {
        "timestamp": "2024-01-15T14:30:00Z",
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.1",
        "source_port": "invalid_port",  # Type invalide
        "destination_port": 80,
        "protocol": "TCP",
        "payload_size": 1024,
        "user_agent": "Mozilla/5.0",
        "status": "success"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers=HEADERS,
        json=invalid_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def main():
    """Fonction principale pour exécuter tous les tests"""
    print("🚀 Démarrage des tests de l'API Flask...")
    print("=" * 50)
    
    try:
        test_health_check()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_data_management()
        test_error_handling()
        
        print("✅ Tous les tests ont été exécutés avec succès!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Erreur: Impossible de se connecter à l'API")
        print("Assurez-vous que l'API est démarrée avec:")
        print("python scripts/start_flask_api.py")
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")


if __name__ == "__main__":
    main()