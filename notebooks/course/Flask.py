#!/usr/bin/env python
# coding: utf-8

# Flask est un **micro-framework open source de développement Web** qui permet de créer des API REST en Python. Micro ne signifie pas qu'il s'agit d'un framework léger, mais Flask se concentre sur la seule tâche de développement web : toutes les couches supplémentaires sont ensuite gérés par les développeurs.
# 
# <img src="https://dv495y1g0kef5.cloudfront.net/training/data_engineer_uber/img/flask.png" />
# 
# <blockquote><p>🙋 <b>Ce que nous allons faire</b></p>
# <ul>
#     <li>Construire l'API de panier utilisateur avec Flask</li>
#     <li>Exécuter l'API en local et interagir avec</li>
# </ul>
# </blockquote>
# 
# <img src="https://media.giphy.com/media/3og0IAQG2BtR13joe4/giphy.gif" />

# In[ ]:


import os
import signal
import subprocess
import time

server = None

def stop_server():
    if server:
        os.killpg(server.pid, signal.SIGTERM)
        server.terminate()

def start_server():
    # Petite astuce pour exécuter le serveur sans quitter le notebook
    stop_server()
    time.sleep(1.5)
    print("Serveur prêt")
    return subprocess.Popen("FLASK_APP=/tmp/server.py flask run", shell=True, preexec_fn=os.setsid)

# ## Un premier exemple
# 
# Une des principales forces de Flask est la possibilité de créer une API en seulement quelques lignes. Examinons le code suivant :

# In[ ]:


%%writefile /tmp/server.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Coucou !"

# La ligne `app = Flask(__name__)` permet d'instancier une nouvelle application Flask, qui fait référence ici à notre API. Ainsi, les routes seront définies à partir de la variable `app`.
# 
# Ensuite, nous définissons la fonction `hello_world` qui retourne simplement la chaîne de caractère `Coucou !`. En ajoutant le décorateur `@app.route('/')` à cette fonction, cela permet de spécifier à l'application Flask que cette fonction sera exécutée sur la route `/`, et la valeur retournée par cette fonction sera par la suite renvoyée au client qui aura envoyé la requête.
# 
# Démarrons le serveur.

# In[ ]:


server = start_server()

# Par défaut, Flask écoute sur l'adresse `localhost` et sur le port $5000$. Ainsi, une requête GET sur l'adresse `127.0.0.1:5000` devrait retourner un code 200 avec le message `Coucou !`.

# In[ ]:


!pip install requests -q
import requests

requests.get("http://127.0.0.1:5000").content

# Flask détecte que nous effectuons une requête sur la route `/`, exécute donc la fonction `hello_world` et retourne le résultat de la fonction au client. Créons une nouvelle route `/cart` qui va pour l'instant renvoyer `Panier vide`.

# In[ ]:


%%writefile /tmp/server.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Coucou !"

@app.route('/cart')
def cart():
    return "Panier vide !"

# In[ ]:


server = start_server()

# Ici, nous définissons une nouvelle route `/cart`, qui retourne également une chaîne de caractère par défaut.

# In[ ]:


requests.get("http://127.0.0.1:5000/cart").content

# ## Création d'un panier virtuel
# 
# Exploitons pleinement le potentiel des API REST. Nous allons à présent appliquer l'exemple que nous avons déroulé sur les API, à savoir la gestion d'un panier d'achat permettant de lister, ajouter, modifier ou supprimer des produits.
# 
# Pour recevoir ou envoyer des données dans une API REST, le format privilégié est le JSON, puisque ce format non structurée n'impose pas de schéma particulier et permet à chaque requête de retourner des données qui lui sont propres. Flask dispose d'un module `jsonify` qui permet d'encoder une liste ou un dictionnaire directement au format JSON.

# In[ ]:


%%writefile /tmp/server.py
from flask import Flask, request, jsonify

app = Flask(__name__)

cart = []

@app.route('/')
def hello_world():
    return "Coucou !"

@app.route('/cart', methods=['GET'])
def list_cart():
    return jsonify(cart), 200

@app.route('/cart', methods=['POST'])
def add_to_cart():
    try:
        body = request.get_json()
        # On s'assure que les champs 'id' et 'quantity' sont bien présents dans le corps de la requête
        if 'id' not in body.keys() or 'quantity' not in body.keys():
            return jsonify({'error': "Missing fields."}), 400
        # Si le produit existe déjà : rajouter la nouvelle quantité à la précédente
        for i, item in enumerate(cart):
            if item['id'] == body['id']:
                cart[i]['quantity'] += int(body['quantity'])
                return jsonify({}), 200
            
        # Si l'on atteint cette partie, alors le produit n'existait pas déjà
        cart.append(body)
        return jsonify({}), 200      
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Détaillons les fonctions `list_cart` et `add_to_cart`. Ces deux fonctions sont liées à la même route `/cart`, la principale différence réside dans le verbe HTTP : la fonction `list_cart` sera exécutée dans le cas d'une méthode GET, alors que la fonction `add_to_cart` sera exécutée dans le cas d'une méthode POST. Cela permet donc d'avoir une seule route mais qui concentre plusieurs fonctionnalités que l'on peut choisir par le verbe d'action.
# 
# La variable `cart` est une liste qui contiendra les produits, où chaque produit est représenté par un **dictionnaire** qui contient deux champs : un champ `id` qui est un identifiant unique du produit dans la base de données, et un champ `quantity` qui précise la quantité associée à ce produit.
# 
# Si un utilisateur souhaite ajouter un produit à son panier, il devra exécuter une méthode POST sur la route `/cart` en fournissant également un corps du message qui est le suivant :
{
    'id': "je8zng",
    'quantity': 1
}
# Dès cette étape, il y a plusieurs actions à entreprendre.
# 
# - Tout d'abord, il faut s'assurer que les champs nécessaires sont bien présents dans le corps du message, à savoir les champs `id` et `quantity`. Il n'est pas possible d'ajouter un produit si l'on ne connait pas son identifiant ou la quantité associée.
# - Ensuite, il faut étudier si le produit n'existe pas déjà dans le panier. Si c'est le cas, il faudra rajouter à la quantité existante celle qui est proposée dans le corps de la requête.
# - Enfin, si le produit n'existe pas déjà dans le panier, il suffit d'ajouter le corps de la requête dans le panier.
# 
# La fonction `check_fields` permet de s'assurer que tous les paramètres requis sont bien présents.

# In[ ]:


def check_fields(body, fields):
    # On récupère les champs requis au format 'ensemble'
    required_parameters_set = set(fields)
    # On récupère les champs du corps de la requête au format 'ensemble'
    fields_set = set(body.keys())
    # Si l'ensemble des champs requis n'est pas inclut dans l'ensemble des champs du corps de la requête
    # Alors s'il manque des paramètres et la valeur False sera renvoyée
    return required_parameters_set <= fields_set

print(check_fields({}, {'id', 'quantity'}))  # Pas bon
print(check_fields({'id': 0}, {'id', 'quantity'}))  # Pas bon
print(check_fields({'id': 0, 'quantity': 0}, {'id', 'quantity'}))  # OK
print(check_fields({'id': 0, 'quantity': 0, 'description': ""}, {'id', 'quantity'}))  # OK

# In[ ]:


%%writefile /tmp/server.py
from flask import Flask, request, jsonify

app = Flask(__name__)

cart = []

def check_fields(body, fields):
    # On récupère les champs requis au format 'ensemble'
    required_parameters_set = set(fields)
    # On récupère les champs du corps de la requête au format 'ensemble'
    fields_set = set(body.keys())
    # Si l'ensemble des champs requis n'est pas inclut dans l'ensemble des champs du corps de la requête
    # Alors s'il manque des paramètres et la valeur False sera renvoyée
    return required_parameters_set <= fields_set

@app.route('/')
def hello_world():
    return "Coucou !"

@app.route('/cart', methods=['GET'])
def list_cart():
    return jsonify(cart), 200

@app.route('/cart', methods=['POST'])
def add_to_cart():
    try:
        body = request.get_json()
        if not check_fields(body, {'id', 'quantity'}):
            # S'il manque un paramètre on retourne une erreur 400
            return jsonify({'error': "Missing fields."}), 400
        
        # On vérifie si le produit n'existe pas déjà
        for i, item in enumerate(cart):
            if item['id'] == body['id']:
                # On a retrouvé ce produit dans le panier, on ajoute à la quantité existante
                cart[i]['quantity'] += int(body['quantity'])
                # On retourne un code 200 pour signaler que tout s'est bien passé
                return jsonify({}), 200
            
         # Si l'on atteint cette partie, alors le produit n'existait pas déjà
        cart.append(body)
        return jsonify({}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# In[ ]:


server = start_server()

# Testons une requête en omettant un champ requis.

# In[ ]:


req = requests.post("http://127.0.0.1:5000/cart", json={
    'id': "je8zng"
})
print(req.status_code, req.json())

# Nous avons bien une erreur 400 puisque un ou plusieurs champs requis sont manquants.

# In[ ]:


req = requests.post("http://127.0.0.1:5000/cart", json={
    'id': "je8zng",
    'quantity': 1
})
print(req.status_code, req.json())

# Tout s'est bien passé, donc en listant les produits du panier, celui que l'on vient d'ajouter est présent.

# In[ ]:


requests.get("http://127.0.0.1:5000/cart").json()

# Si, à présent, nous rajoutons un même produit, le comportement de l'API que nous avons mis en place devrait rajouter la quantité au produit déjà présent dans le panier.

# In[ ]:


req = requests.post("http://127.0.0.1:5000/cart", json={
    'id': "je8zng",
    'quantity': 2
})
print(req.status_code, req.json())

# Le produit d'identifiant `je8zng` devrait donc apparaître avec une quantité de 3.

# In[ ]:


requests.get("http://127.0.0.1:5000/cart").json()

# ### Méthode PATCH
# 
# Intégrons dorénavant la possibilité de mettre à jour la quantité pour un produit. Pour cela, l'architecture REST préconise d'utiliser le verbe PATCH pour mettre à jour une ressource. La fonction `edit_cart` déclenchera les instructions pour modifier les quantités d'un produit.

# In[ ]:


%%writefile /tmp/server.py
from flask import Flask, request, jsonify

app = Flask(__name__)

cart = [{
    'id': "je8zng",
    'quantity': 3
}]

def check_fields(body, fields):
    # On récupère les champs requis au format 'ensemble'
    required_parameters_set = set(fields)
    # On récupère les champs du corps de la requête au format 'ensemble'
    fields_set = set(body.keys())
    # Si l'ensemble des champs requis n'est pas inclut dans l'ensemble des champs du corps de la requête
    # Alors s'il manque des paramètres et la valeur False sera renvoyée
    return required_parameters_set <= fields_set

@app.route('/')
def hello_world():
    return "Coucou !"

@app.route('/cart', methods=['GET'])
def list_cart():
    return jsonify(cart), 200

@app.route('/cart', methods=['POST'])
def add_to_cart():
    try:
        body = request.get_json()
        if not check_fields(body, {'id', 'quantity'}):
            # S'il manque un paramètre on retourne une erreur 400
            return jsonify({'error': "Missing fields."}), 400
        
        # On vérifie si le produit n'existe pas déjà
        for i, item in enumerate(cart):
            if item['id'] == body.get('id', ""):
                # On a retrouvé ce produit dans le panier, on ajoute à la quantité existante
                cart[i]['quantity'] += int(body.get('quantity', 0))
                # On retourne un code 200 pour signaler que tout s'est bien passé
                return jsonify({}), 200
            
         # Si l'on atteint cette partie, alors le produit n'existait pas déjà
        cart.append(body)
        return jsonify({}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cart', methods=['PATCH'])
def edit_cart():
    try:
        body = request.get_json()
        if not check_fields(body, {'id', 'quantity'}):
            # S'il manque un paramètre on retourne une erreur 400
            return jsonify({'error': "Missing fields."}), 400

        for i, item in enumerate(cart):
            if item['id'] == body['id']:
                # On met à jour la quantité
                cart[i]['quantity'] = int(body['quantity'])
                return jsonify({}), 200
        
        # Si l'on atteint cette partie, alors le produit n'existait pas : on ne peut pas mettre à jour !
        return jsonify({'error': "Product not found."}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# In[ ]:


server = start_server()

# In[ ]:


req = requests.patch("http://127.0.0.1:5000/cart", json={
    'id': "aaaaa",
    'quantity': 10
})
print(req.status_code, req.json())

# In[ ]:


req = requests.patch("http://127.0.0.1:5000/cart", json={
    'id': "je8zng",
    'quantity': 10
})
print(req.status_code, req.json())

# In[ ]:


requests.get("http://127.0.0.1:5000/cart").json()

# ### Méthode DELETE
# 
# La dernière fonctionnalité est la suppression d'un produit dans le panier : il n'y a donc plus besoin que le champ `quantity` soit présent.

# In[ ]:


%%writefile /tmp/server.py
from flask import Flask, request, jsonify

app = Flask(__name__)

cart = [{
    'id': "je8zng",
    'quantity': 3
}]

def check_fields(body, fields):
    # On récupère les champs requis au format 'ensemble'
    required_parameters_set = set(fields)
    # On récupère les champs du corps de la requête au format 'ensemble'
    fields_set = set(body.keys())
    # Si l'ensemble des champs requis n'est pas inclut dans l'ensemble des champs du corps de la requête
    # Alors s'il manque des paramètres et la valeur False sera renvoyée
    return required_parameters_set <= fields_set

@app.route('/')
def hello_world():
    return "Coucou !"

@app.route('/cart', methods=['GET'])
def list_cart():
    return jsonify(cart), 200

@app.route('/cart', methods=['POST'])
def add_to_cart():
    try:
        body = request.get_json()
        if not check_fields(body, {'id', 'quantity'}):
            # S'il manque un paramètre on retourne une erreur 400
            return jsonify({'error': "Missing fields."}), 400
        
        # On vérifie si le produit n'existe pas déjà
        for i, item in enumerate(cart):
            if item['id'] == body.get('id', ""):
                # On a retrouvé ce produit dans le panier, on ajoute à la quantité existante
                cart[i]['quantity'] += int(body.get('quantity', 0))
                # On retourne un code 200 pour signaler que tout s'est bien passé
                return jsonify({}), 200
            
         # Si l'on atteint cette partie, alors le produit n'existait pas déjà
        cart.append(body)
        return jsonify({}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cart', methods=['PATCH'])
def edit_cart():
    try:
        body = request.get_json()
        if not check_fields(body, {'id', 'quantity'}):
            # S'il manque un paramètre on retourne une erreur 400
            return jsonify({'error': "Missing fields."}), 400

        for i, item in enumerate(cart):
            if item['id'] == body.get('id', ""):
                # On met à jour la quantité
                cart[i]['quantity'] = int(body.get('quantity', 0))
                return jsonify({}), 200
        
        # Si l'on atteint cette partie, alors le produit n'existait pas : on ne peut pas mettre à jour !
        return jsonify({'error': "Product not found."}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cart', methods=['DELETE'])
def remove_from_cart():
    try:
        body = request.get_json()
        if not check_fields(body, {'id'}):
            # S'il manque un paramètre on retourne une erreur 400
            return jsonify({'error': "Missing fields."}), 400
        
        for i, item in enumerate(cart):
            if item['id'] == body['id']:
                # On supprime le produit du panier
                del cart[i]
                return jsonify({}), 200
            
        # Si l'on atteint cette partie, alors le produit n'existait pas : on ne peut pas supprimer !
        return jsonify({'error': "Product not found."}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# In[ ]:


server = start_server()

# In[ ]:


req = requests.delete("http://127.0.0.1:5000/cart", json={
    'id': "je8zng"
})
print(req.status_code, req.json())

# In[ ]:


requests.get("http://127.0.0.1:5000/cart").json()

# Une fois l'expérimentation terminée, nous pouvons stopper le serveur.

# In[ ]:


stop_server()

# ## ✔️ Conclusion
# 
# L'intégration de l'architecture REST se fait sans difficultés avec Flask.
# 
# - Nous avons vu qu'il est très facile de construire une API REST.
# - Nous avons pu interagir avec l'API et y ajouter des fonctionnalités.
# 
# > ➡️ La dernière étape consiste à intégrer le modèle de Machine Learning dans l'API pour pouvoir calculer des prédictions depuis un serveur distant.
