# API Flask - Network Intrusion Detection

API REST complète pour la détection d'intrusion réseau basée sur le projet Kedro.

## 🚀 Démarrage rapide

```bash
# Installer les dépendances
uv sync

# Démarrer l'API
python scripts/start_flask_api.py
```

L'API sera accessible sur `http://127.0.0.1:5000`

## 📚 Documentation

- **Swagger UI** : `http://127.0.0.1:5000/api/docs`
- **Health Check** : `http://127.0.0.1:5000/health`

## 🛠️ Endpoints

### 🏥 Health
- `GET /health` - Vérification de l'état de l'API

### 📊 Data Management (CRUD)
- `GET /data` - Lister toutes les données d'intrusion
- `POST /data` - Ajouter une nouvelle donnée
- `PATCH /data` - Modifier une donnée existante
- `DELETE /data` - Supprimer une donnée

### 🤖 Machine Learning
- `POST /predict` - Prédiction individuelle
- `POST /predict/batch` - Prédictions par lot
- `GET /model/info` - Informations sur le modèle

## 📝 Exemples d'utilisation

### Prédiction individuelle
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T14:30:00Z",
    "source_ip": "192.168.1.100",
    "destination_ip": "10.0.0.1",
    "source_port": 12345,
    "destination_port": 22,
    "protocol": "TCP",
    "payload_size": 0,
    "user_agent": "bot-scanner",
    "status": "failed"
  }'
```

### Ajout de données
```bash
curl -X POST http://127.0.0.1:5000/data \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T14:30:00Z",
    "source_ip": "192.168.1.100",
    "destination_ip": "10.0.0.1",
    "source_port": 12345,
    "destination_port": 80,
    "protocol": "TCP",
    "payload_size": 1024,
    "user_agent": "Mozilla/5.0",
    "status": "success"
  }'
```

## 🧪 Tests

```bash
# Lancer les tests
uv run pytest tests/test_api.py -v

# Avec couverture
uv run pytest tests/test_api.py --cov=src/kedro_project/api
```

## 📁 Structure

```
src/kedro_project/api/
├── __init__.py          # Module API
├── app.py              # Application Flask principale
├── routes.py           # Routes ML et prédictions
├── models.py           # Modèles de validation
├── utils.py            # Fonctions utilitaires
├── swagger.json        # Documentation OpenAPI
└── run_server.py       # Script de démarrage
```

## ⚙️ Configuration

### Constantes de prédiction
- `INTRUSION_THRESHOLD = 0.5` - Seuil de détection d'intrusion
- `LARGE_PAYLOAD_THRESHOLD = 10000` - Seuil payload suspect
- `LOW_RISK_THRESHOLD = 0.3` - Seuil risque faible
- `MEDIUM_RISK_THRESHOLD = 0.7` - Seuil risque moyen

### Ports
- `MAX_PORT = 65535` - Port maximum valide
- `MIN_PORT = 0` - Port minimum valide

## 🔧 Modèle de prédiction

Le modèle actuel utilise des **règles heuristiques** basées sur :
- Ports suspects (22, 23, 135, 139, 445, 1433, 3389, 5432)
- Taille de payload anormale (>10KB ou vide)
- Statuts d'erreur (failed, error, timeout)
- User agents suspects (bot, crawler, scanner, exploit)

## 📊 Format des données

### Données d'entrée
```json
{
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
```

### Réponse de prédiction
```json
{
  "prediction": {
    "is_intrusion": false,
    "confidence": 0.2,
    "risk_level": "low"
  },
  "input_data": { ... },
  "model_version": "heuristic_v1.0"
}
```

## 🚧 Évolutions futures

- Intégration avec MLflow pour modèles ML réels
- Authentification et autorisation
- Rate limiting
- Logging avancé
- Métriques et monitoring
- Support de bases de données persistantes