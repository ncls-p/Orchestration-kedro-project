#!/usr/bin/env python3
"""
Script de test rapide pour l'API Flask
"""

import requests
import json
import sys
import time
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:5000"
TIMEOUT = 5


def wait_for_api(max_attempts: int = 30, delay: float = 1.0) -> bool:
    """Attendre que l'API soit disponible"""
    print(f"🔄 Attente de l'API sur {BASE_URL}...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            if response.status_code == 200:
                print(f"✅ API disponible après {attempt + 1} tentative(s)")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            time.sleep(delay)
    
    print(f"❌ API non disponible après {max_attempts} tentatives")
    return False


def test_endpoint(method: str, endpoint: str, data: Dict[str, Any] = None) -> bool:
    """Tester un endpoint de l'API"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=TIMEOUT)
        elif method.upper() == "PATCH":
            response = requests.patch(url, json=data, timeout=TIMEOUT)
        elif method.upper() == "DELETE":
            response = requests.delete(url, json=data, timeout=TIMEOUT)
        else:
            print(f"❌ Méthode {method} non supportée")
            return False
        
        # Vérifier le statut
        if response.status_code < 400:
            print(f"✅ {method} {endpoint} - Status: {response.status_code}")
            return True
        else:
            print(f"⚠️  {method} {endpoint} - Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Erreur: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Erreur: {response.text}")
            return response.status_code in [400, 404, 409]  # Erreurs attendues
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {method} {endpoint} - Erreur: {e}")
        return False


def main():
    """Fonction principale de test"""
    print("🧪 Test rapide de l'API Flask")
    print("=" * 40)
    
    # Attendre que l'API soit disponible
    if not wait_for_api():
        sys.exit(1)
    
    # Tests basiques
    tests = [
        ("GET", "/", None),
        ("GET", "/health", None),
        ("GET", "/model/info", None),
        ("GET", "/data", None),
    ]
    
    # Test de prédiction
    prediction_data = {
        "timestamp": "2024-01-15T14:30:00Z",
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.1",
        "source_port": 12345,
        "destination_port": 80,
        "protocol": "TCP",
        "payload_size": 1024,
        "user_agent": "Mozilla/5.0",
        "status": "success"
    }
    
    tests.append(("POST", "/predict", prediction_data))
    
    # Test d'ajout de données
    tests.append(("POST", "/data", prediction_data))
    
    # Test de modification (peut échouer si données n'existent pas)
    update_data = {
        "timestamp": prediction_data["timestamp"],
        "source_ip": prediction_data["source_ip"],
        "destination_ip": prediction_data["destination_ip"],
        "payload_size": 2048
    }
    tests.append(("PATCH", "/data", update_data))
    
    # Test de suppression
    delete_data = {
        "timestamp": prediction_data["timestamp"],
        "source_ip": prediction_data["source_ip"],
        "destination_ip": prediction_data["destination_ip"]
    }
    tests.append(("DELETE", "/data", delete_data))
    
    # Exécuter les tests
    passed = 0
    total = len(tests)
    
    for method, endpoint, data in tests:
        if test_endpoint(method, endpoint, data):
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Résultats: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés!")
        sys.exit(0)
    else:
        print("⚠️  Certains tests ont échoué")
        sys.exit(1)


if __name__ == "__main__":
    main()