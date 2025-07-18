"""Tests pour l'API Flask"""

import pytest
import json
from unittest.mock import patch
from kedro_project.api.app import app
from kedro_project.api.utils import check_fields
from kedro_project.api.models import NetworkIntrusionData

@pytest.fixture
def client():
    """Client de test Flask"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_network_data():
    """Données d'exemple pour les tests"""
    return {
        'timestamp': '2024-01-01T12:00:00Z',
        'source_ip': '192.168.1.100',
        'destination_ip': '10.0.0.1',
        'source_port': 12345,
        'destination_port': 80,
        'protocol': 'TCP',
        'payload_size': 1024,
        'user_agent': 'Mozilla/5.0',
        'status': 'success'
    }

def test_hello_world(client):
    """Test de la route de base"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'API Flask - Kedro Network Intrusion Detection' in response.data

def test_health_check(client):
    """Test du health check"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert data['service'] == 'network-intrusion-detection-api'

def test_check_fields_utility():
    """Test de la fonction check_fields"""
    # Test avec tous les champs présents
    assert check_fields({'id': 1, 'quantity': 2}, {'id', 'quantity'}) == True
    
    # Test avec des champs manquants
    assert check_fields({'id': 1}, {'id', 'quantity'}) == False
    
    # Test avec des champs supplémentaires
    assert check_fields({'id': 1, 'quantity': 2, 'extra': 3}, {'id', 'quantity'}) == True
    
    # Test avec body vide
    assert check_fields({}, {'id', 'quantity'}) == False
    
    # Test avec body None
    assert check_fields(None, {'id', 'quantity'}) == False

def test_network_intrusion_data_validation(sample_network_data):
    """Test de la validation des données d'intrusion"""
    # Test avec des données valides
    intrusion_data = NetworkIntrusionData(sample_network_data)
    assert intrusion_data.validate_required_fields() == True
    assert intrusion_data.validate_data_types() == True
    
    # Test avec des données invalides - port négatif
    invalid_data = sample_network_data.copy()
    invalid_data['source_port'] = -1
    intrusion_data = NetworkIntrusionData(invalid_data)
    assert intrusion_data.validate_data_types() == False
    
    # Test avec des données invalides - port trop grand
    invalid_data = sample_network_data.copy()
    invalid_data['destination_port'] = 70000
    intrusion_data = NetworkIntrusionData(invalid_data)
    assert intrusion_data.validate_data_types() == False
    
    # Test avec des données invalides - payload_size négatif
    invalid_data = sample_network_data.copy()
    invalid_data['payload_size'] = -100
    intrusion_data = NetworkIntrusionData(invalid_data)
    assert intrusion_data.validate_data_types() == False

def test_add_data_success(client, sample_network_data):
    """Test d'ajout de données valides"""
    response = client.post('/data', 
                          json=sample_network_data,
                          content_type='application/json')
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['message'] == 'Data added successfully'

def test_add_data_missing_fields(client):
    """Test d'ajout de données avec des champs manquants"""
    invalid_data = {
        'timestamp': '2024-01-01T12:00:00Z',
        'source_ip': '192.168.1.100'
        # Champs manquants
    }
    response = client.post('/data', 
                          json=invalid_data,
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Missing required fields' in data['error']

def test_add_data_invalid_types(client):
    """Test d'ajout de données avec des types invalides"""
    invalid_data = {
        'timestamp': '2024-01-01T12:00:00Z',
        'source_ip': '192.168.1.100',
        'destination_ip': '10.0.0.1',
        'source_port': 'invalid_port',  # Type invalide
        'destination_port': 80,
        'protocol': 'TCP',
        'payload_size': 1024,
        'user_agent': 'Mozilla/5.0',
        'status': 'success'
    }
    response = client.post('/data', 
                          json=invalid_data,
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Invalid data types' in data['error']

def test_list_data_empty(client):
    """Test de listage des données quand vide"""
    response = client.get('/data')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_predict_success(client, sample_network_data):
    """Test de prédiction avec des données valides"""
    response = client.post('/predict', 
                          json=sample_network_data,
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'is_intrusion' in data['prediction']
    assert 'confidence' in data['prediction']
    assert 'risk_level' in data['prediction']

def test_predict_missing_fields(client):
    """Test de prédiction avec des champs manquants"""
    invalid_data = {
        'timestamp': '2024-01-01T12:00:00Z',
        'source_ip': '192.168.1.100'
        # Champs manquants
    }
    response = client.post('/predict', 
                          json=invalid_data,
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Missing required fields' in data['error']

def test_predict_batch_success(client, sample_network_data):
    """Test de prédiction par lot"""
    batch_data = {
        'data': [sample_network_data, sample_network_data.copy()]
    }
    response = client.post('/predict/batch', 
                          json=batch_data,
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'predictions' in data
    assert len(data['predictions']) == 2
    assert data['total_processed'] == 2

def test_model_info(client):
    """Test des informations du modèle"""
    response = client.get('/model/info')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'model_name' in data
    assert 'model_version' in data
    assert 'features' in data
    assert 'output' in data

def test_delete_data_success(client, sample_network_data):
    """Test de suppression de données"""
    # D'abord ajouter les données
    client.post('/data', json=sample_network_data, content_type='application/json')
    
    # Puis les supprimer
    delete_data = {
        'timestamp': sample_network_data['timestamp'],
        'source_ip': sample_network_data['source_ip'],
        'destination_ip': sample_network_data['destination_ip']
    }
    response = client.delete('/data', 
                           json=delete_data,
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['message'] == 'Data deleted successfully'

def test_delete_data_not_found(client):
    """Test de suppression de données inexistantes"""
    delete_data = {
        'timestamp': '2024-01-01T12:00:00Z',
        'source_ip': '192.168.1.999',
        'destination_ip': '10.0.0.999'
    }
    response = client.delete('/data', 
                           json=delete_data,
                           content_type='application/json')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'Data not found' in data['error']