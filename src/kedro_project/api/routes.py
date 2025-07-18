"""Routes pour les prédictions ML"""

from flask import jsonify, request

from .models import NetworkIntrusionData
from .utils import check_fields

# Constantes pour la prédiction
INTRUSION_THRESHOLD = 0.5
LARGE_PAYLOAD_THRESHOLD = 10000
LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.7


def create_prediction_routes(app):
    """Crée les routes pour les prédictions ML"""

    @app.route("/predict", methods=["POST"])
    def predict_intrusion():
        """Prédit si une connexion réseau est une intrusion"""
        try:
            body = request.get_json()

            # Vérifier les champs requis
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
                return jsonify(
                    {"error": "Missing required fields for prediction."}
                ), 400

            # Valider les données
            intrusion_data = NetworkIntrusionData(body)
            if not intrusion_data.validate_data_types():
                return jsonify({"error": "Invalid data types or values."}), 400

            # Simulation de prédiction (à remplacer par l'intégration MLflow)
            prediction_data = intrusion_data.to_prediction_format()

            # Logique de prédiction simple basée sur des règles heuristiques
            prediction_score = calculate_intrusion_score(prediction_data)
            is_intrusion = prediction_score > INTRUSION_THRESHOLD

            return jsonify(
                {
                    "prediction": {
                        "is_intrusion": is_intrusion,
                        "confidence": prediction_score,
                        "risk_level": get_risk_level(prediction_score),
                    },
                    "input_data": prediction_data,
                    "model_version": "heuristic_v1.0",
                }
            ), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        """Prédit pour un lot de connexions réseau"""
        try:
            body = request.get_json()

            if not check_fields(body, {"data"}):
                return jsonify(
                    {
                        "error": "Missing 'data' field containing list of network connections."
                    }
                ), 400

            if not isinstance(body["data"], list):
                return jsonify({"error": "'data' field must be a list."}), 400

            predictions = []

            for i, item in enumerate(body["data"]):
                try:
                    # Valider chaque élément
                    intrusion_data = NetworkIntrusionData(item)
                    if not intrusion_data.validate_required_fields():
                        predictions.append(
                            {
                                "index": i,
                                "error": "Missing required fields",
                                "prediction": None,
                            }
                        )
                        continue

                    if not intrusion_data.validate_data_types():
                        predictions.append(
                            {
                                "index": i,
                                "error": "Invalid data types",
                                "prediction": None,
                            }
                        )
                        continue

                    # Prédiction
                    prediction_data = intrusion_data.to_prediction_format()
                    prediction_score = calculate_intrusion_score(prediction_data)
                    is_intrusion = prediction_score > INTRUSION_THRESHOLD

                    predictions.append(
                        {
                            "index": i,
                            "prediction": {
                                "is_intrusion": is_intrusion,
                                "confidence": prediction_score,
                                "risk_level": get_risk_level(prediction_score),
                            },
                            "error": None,
                        }
                    )

                except Exception as e:
                    predictions.append(
                        {"index": i, "error": str(e), "prediction": None}
                    )

            return jsonify(
                {
                    "predictions": predictions,
                    "total_processed": len(body["data"]),
                    "model_version": "heuristic_v1.0",
                }
            ), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/model/info", methods=["GET"])
    def model_info():
        """Informations sur le modèle utilisé"""
        return jsonify(
            {
                "model_name": "Network Intrusion Detection",
                "model_version": "heuristic_v1.0",
                "model_type": "Rule-based heuristic",
                "features": [
                    "timestamp",
                    "source_ip",
                    "destination_ip",
                    "source_port",
                    "destination_port",
                    "protocol",
                    "payload_size",
                    "user_agent",
                    "status",
                ],
                "output": {
                    "is_intrusion": "boolean",
                    "confidence": "float (0-1)",
                    "risk_level": "string (low/medium/high)",
                },
                "description": "Heuristic-based intrusion detection model for network traffic analysis",
            }
        ), 200


def calculate_intrusion_score(data):
    """Calcule un score d'intrusion basé sur des règles heuristiques"""
    score = 0.0

    # Ports suspects
    suspicious_ports = {22, 23, 135, 139, 445, 1433, 3389, 5432}
    if data.get("destination_port") in suspicious_ports:
        score += 0.3

    # Taille de payload suspecte
    payload_size = data.get("payload_size", 0)
    if payload_size > LARGE_PAYLOAD_THRESHOLD:  # Payload très large
        score += 0.2
    elif payload_size == 0:  # Payload vide
        score += 0.1

    # Protocoles suspects
    if data.get("protocol", "").lower() in ["tcp", "udp"]:
        if data.get("status", "").lower() in ["failed", "error", "timeout"]:
            score += 0.3

    # User Agent suspects
    user_agent = data.get("user_agent", "").lower()
    if any(
        suspect in user_agent for suspect in ["bot", "crawler", "scanner", "exploit"]
    ):
        score += 0.4

    # Normaliser le score entre 0 et 1
    return min(score, 1.0)


def get_risk_level(score):
    """Détermine le niveau de risque basé sur le score"""
    if score < LOW_RISK_THRESHOLD:
        return "low"
    elif score < MEDIUM_RISK_THRESHOLD:
        return "medium"
    else:
        return "high"
