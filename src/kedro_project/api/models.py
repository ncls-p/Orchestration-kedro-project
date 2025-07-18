"""Modèles de validation pour l'API Flask"""

from typing import Any

# Constantes pour la validation
MAX_PORT = 65535
MIN_PORT = 0


class NetworkIntrusionData:
    """
    Classe pour valider les données d'intrusion réseau
    """

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def validate_required_fields(self) -> bool:
        """Valide que tous les champs requis sont présents"""
        required_fields = {
            "timestamp",
            "source_ip",
            "destination_ip",
            "source_port",
            "destination_port",
            "protocol",
            "payload_size",
            "user_agent",
            "status",
        }

        return all(field in self.data for field in required_fields)

    def validate_data_types(self) -> bool:
        """Valide les types de données"""
        try:
            # Validation des types de base
            validations = [
                isinstance(self.data.get("source_port"), int),
                isinstance(self.data.get("destination_port"), int),
                isinstance(self.data.get("payload_size"), int | float),
                isinstance(self.data.get("protocol"), str),
                isinstance(self.data.get("status"), str),
            ]

            if not all(validations):
                return False

            # Validation des plages
            source_port = self.data.get("source_port")
            destination_port = self.data.get("destination_port")
            payload_size = self.data.get("payload_size")

            # Validation des ports
            if source_port is not None and not (MIN_PORT <= source_port <= MAX_PORT):
                return False
            if destination_port is not None and not (MIN_PORT <= destination_port <= MAX_PORT):
                return False

            # Validation du payload
            if payload_size is not None and payload_size < 0:
                return False

            return True
        except (TypeError, ValueError):
            return False

    def to_prediction_format(self) -> dict[str, Any]:
        """Convertit les données au format attendu par le modèle"""
        return {
            "timestamp": self.data["timestamp"],
            "source_ip": self.data["source_ip"],
            "destination_ip": self.data["destination_ip"],
            "source_port": self.data["source_port"],
            "destination_port": self.data["destination_port"],
            "protocol": self.data["protocol"],
            "payload_size": self.data["payload_size"],
            "user_agent": self.data["user_agent"],
            "status": self.data["status"],
        }
