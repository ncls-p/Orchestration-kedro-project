"""Utilitaires pour l'API Flask"""


def check_fields(body, fields):
    """
    Vérifie que tous les champs requis sont présents dans le corps de la requête.

    Args:
        body (dict): Corps de la requête JSON
        fields (set): Ensemble des champs requis

    Returns:
        bool: True si tous les champs sont présents, False sinon
    """
    if not body:
        return False

    required_parameters_set = set(fields)
    fields_set = set(body.keys())

    return required_parameters_set <= fields_set
