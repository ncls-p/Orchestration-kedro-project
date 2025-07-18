"""API Flask pour le projet Kedro d'intrusion detection"""

import json
import os

from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint

from .models import NetworkIntrusionData
from .routes import create_prediction_routes
from .utils import check_fields

app = Flask(__name__)

# Configuration
app.config["JSON_SORT_KEYS"] = False
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

# Variable globale pour simuler un stockage en mémoire
data_store = []


@app.route("/")
def hello_world():
    """Route de base pour tester l'API"""
    return """
    <h1>🚀 API Flask - Kedro Network Intrusion Detection</h1>
    <p>L'API est en cours d'exécution !</p>
    <h2>📚 Documentation</h2>
    <ul>
        <li><a href="/api/docs">📖 Documentation Swagger (Interactive)</a></li>
        <li><a href="/health">🏥 Health Check</a></li>
        <li><a href="/model/info">📊 Informations du modèle</a></li>
    </ul>
    <h2>🔗 Endpoints principaux</h2>
    <ul>
        <li><code>GET /data</code> - Lister les données</li>
        <li><code>POST /data</code> - Ajouter des données</li>
        <li><code>POST /predict</code> - Prédiction individuelle</li>
        <li><code>POST /predict/batch</code> - Prédictions par lot</li>
    </ul>
    """


@app.route("/health")
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify(
        {
            "status": "healthy",
            "service": "network-intrusion-detection-api",
            "version": "1.0.0",
        }
    ), 200


@app.route("/data", methods=["GET"])
def list_data():
    """Liste toutes les données d'intrusion stockées"""
    return jsonify(data_store), 200


@app.route("/data", methods=["POST"])
def add_data():
    """Ajoute une nouvelle donnée d'intrusion"""
    try:
        body = request.get_json()

        if not check_fields(
            body,
            {
                "timestamp",
                "source_ip",
                "destination_ip",
                "source_port",
                "destination_port",
                "protocol",
                "payload_size",
                "user_agent",
                "status",
            },
        ):
            return jsonify({"error": "Missing required fields."}), 400

        # Validation des données
        intrusion_data = NetworkIntrusionData(body)
        if not intrusion_data.validate_data_types():
            return jsonify({"error": "Invalid data types or values."}), 400

        # Vérifier si les données existent déjà (par timestamp + source_ip + destination_ip)
        for item in data_store:
            if (
                item.get("timestamp") == body.get("timestamp")
                and item.get("source_ip") == body.get("source_ip")
                and item.get("destination_ip") == body.get("destination_ip")
            ):
                return jsonify({"error": "Data already exists."}), 409

        # Ajouter les données
        data_store.append(body)
        return jsonify({"message": "Data added successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/data", methods=["PATCH"])
def update_data():
    """Met à jour une donnée d'intrusion existante"""
    try:
        body = request.get_json()

        if not check_fields(body, {"timestamp", "source_ip", "destination_ip"}):
            return jsonify(
                {
                    "error": "Missing identification fields (timestamp, source_ip, destination_ip)."
                }
            ), 400

        # Chercher les données à mettre à jour
        for i, item in enumerate(data_store):
            if (
                item.get("timestamp") == body.get("timestamp")
                and item.get("source_ip") == body.get("source_ip")
                and item.get("destination_ip") == body.get("destination_ip")
            ):
                # Mettre à jour uniquement les champs fournis
                for key, value in body.items():
                    if key not in ["timestamp", "source_ip", "destination_ip"]:
                        data_store[i][key] = value

                return jsonify({"message": "Data updated successfully"}), 200

        return jsonify({"error": "Data not found."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/data", methods=["DELETE"])
def remove_data():
    """Supprime une donnée d'intrusion"""
    try:
        body = request.get_json()

        if not check_fields(body, {"timestamp", "source_ip", "destination_ip"}):
            return jsonify(
                {
                    "error": "Missing identification fields (timestamp, source_ip, destination_ip)."
                }
            ), 400

        # Chercher et supprimer les données
        for i, item in enumerate(data_store):
            if (
                item.get("timestamp") == body.get("timestamp")
                and item.get("source_ip") == body.get("source_ip")
                and item.get("destination_ip") == body.get("destination_ip")
            ):
                del data_store[i]
                return jsonify({"message": "Data deleted successfully"}), 200

        return jsonify({"error": "Data not found."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Configuration Swagger UI
SWAGGER_URL = "/api/docs"
API_URL = "/static/swagger.json"


# Servir le fichier swagger.json
@app.route("/static/swagger.json")
def swagger_json():
    """Servir le fichier swagger.json"""
    swagger_file = os.path.join(os.path.dirname(__file__), "swagger.json")
    with open(swagger_file) as f:
        return jsonify(json.load(f))


# Créer le blueprint Swagger UI
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "Kedro Network Intrusion Detection API",
        "dom_id": "#swagger-ui",
        "layout": "BaseLayout",
        "deepLinking": True,
        "showExtensions": True,
        "showCommonExtensions": True,
    },
)

# Enregistrer le blueprint Swagger UI
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Redirect /docs to /api/docs for convenience
@app.route("/docs")
def redirect_to_docs():
    """Redirection vers la documentation Swagger"""
    from flask import redirect
    return redirect("/api/docs")

# Handle favicon.ico requests
@app.route("/favicon.ico")
def favicon():
    """Serve a simple favicon to avoid 404 errors"""
    return "", 204

# Enregistrer les routes de prédiction
create_prediction_routes(app)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
