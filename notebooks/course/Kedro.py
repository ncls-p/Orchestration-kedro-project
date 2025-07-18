#!/usr/bin/env python
# coding: utf-8

# Il est temps d'organiser tout notre pipeline ML, qui est actuellement séparé dans plusieurs Notebooks. Pour nous aider, nous allons utiliser **Kedro**, un outil open-source permettant de créer des projets de Machine Learning reproductibles, maintenables et modulaires (i.e. plusieurs fichiers), le tout sans trop d'efforts. C'est donc un outil sur-mesure pour les ML Engineers !
# 
# <blockquote><p>🙋 <b>Ce que nous allons faire</b></p>
# <ul>
#     <li>Créer un premier projet Kedro et comprendre son architecture</li>
#     <li>Comprendre les concepts de Kedro</li>
#     <li>Construire un premier pipeline de traitement de données</li>
# </ul>
# </blockquote>
# 
# <img src="https://media.giphy.com/media/ZVUu5Pm23hDZS/giphy.gif" width="300" />

# ## Kedro
# 
# Qu'est-ce que Kedro, et pourquoi avons-nous besoin de cet outil ? La méthode classique pour construire des modèles de Machine Learning est d'utiliser Jupyter Notebook. Mais cette méthode n'est pas du tout viable, notamment lorsqu'il s'agit de déployer le modèle en production dans un futur proche. Face à cette situation, on préfère donc construire un projet entier, dont les Notebooks sont en réalité des phases de recherche, d'expérimentation mais ne constituent pas en soi le coeur de sujet du projet. Dès lors que l'on met en place une architecture de code source, il est nécessaire d'adopter de bonnes pratiques, aussi bien héritées des environnements IT que celles utilisées par les Data Scientists.
# 
# <img src="https://repository-images.githubusercontent.com/182067506/4c724a00-48f4-11ea-84a5-cf8292b07d8e" width="600">
# 
# Kedro a été développé pour appliquer ces bonnes pratiques tout au long d'un projet de Machine Learning.
# 
# - Éviter au maximum de dépendre de Jupyter Notebooks qui empêche la production d'un code source maintenable et reproductible.
# - Améliorer la collaboration entre les différents acteurs (Data Scientists, Data Engineers, DevOps) aux compétences diverses dans un projet.
# - Augmenter l'efficacité en appliquant la modularité du code, les séparations entre les données et leur utilisation ou encore en optimisant les exécutions de traitements atomiques.
# 
# En bref, Kedro nous permet d'avoir un projet Python **entièrement pensé** pour le Machine Learning et optimisé dans ce sens. Il existe d'autres alternatives à Kedro (comme Kubeflow qui se base sur Kubernetes), mais il a l'avantage d'être rapide à prendre en main et possède une communauté déjà active.
# 
# ### Premiers pas avec Kedro
# 
# Essayons de créer un premier projet avec Kedro que nous allons nommer `purchase-predict`.
# 
# <div class="alert alert-danger">
#     La version que nous utiliserons est 0.17.0 : en utilisant une version plus récente, il se peut que des erreurs de compatibilité surviennent. Il est donc conseillé d'utiliser la version 0.17.0 en local pour suivre le cours.
# </div>
# 
# Créons un nouveau terminal et un nouveau projet Kedro.
kedro new
# Il nous est demandé un nom de projet. Nous laissons ensuite les autres informations vides (la valeur par défaut est affichée entre crochets).
Project Name:
=============
Please enter a human readable name for your new project.
Spaces and punctuation are allowed.
 [New Kedro Project]: purchase-predict

Repository Name:
================
Please enter a directory name for your new project repository.
Alphanumeric characters, hyphens and underscores are allowed.
Lowercase is recommended.
 [purchase-predict]: 

Python Package Name:
====================
Please enter a valid Python package name for your project package.
Alphanumeric characters and underscores are allowed.
Lowercase is recommended. Package name must start with a letter or underscore.
 [purchase_predict]: 
# Avec cette commande, Kedro génère le dossier `purchase-predict` et y configure une architecture par défaut.
├── conf
│   ├── base
│   │   ├── catalog.yml
│   │   ├── credentials.yml
│   │   ├── logging.yml
│   │   └── parameters.yml
│   ├── local
│   └── README.md
├── data
│   ├── 01_raw
│   ├── 02_intermediate
│   ├── 03_primary
│   ├── 04_feature
│   ├── 05_model_input
│   ├── 06_models
│   ├── 07_model_output
│   └── 08_reporting
├── docs
│   └── source
│       ├── conf.py
│       └── index.rst
├── logs
│   └── journals
├── notebooks
├── pyproject.toml
├── README.md
├── setup.cfg
└── src
    ├── purchase_predict
    │   ├── cli.py
    │   ├── hooks.py
    │   ├── __init__.py
    │   ├── pipelines
    │   │   └── __init__.py
    │   ├── run.py
    │   └── settings.py
    ├── requirements.txt
    ├── setup.py
    └── tests
        ├── __init__.py
        ├── pipelines
        │   └── __init__.py
        └── test_run.py
# Détaillons tout d'abord chaque dossier de premier niveau.
# 
# - `conf` contient tous les fichiers de configuration des paramètres (code, modèle) ainsi que les clés et secrets nécessaires.
# - `data` contient plusieurs dossiers qui correspondent aux données utilisées ou produits à chaque étape du pipeline (base d'apprentissage, matrices des *features*, modèle sérialisé).
# - `docs` contient des fichiers de documentation.
# - `logs` contient les journaux d'événements de Kedro.
# - `notebooks` permet de stocker des notebooks.
# - `src` contient tous les codes nécessaire pour créer les pipelines.
# 
# C'est notamment dans le dossier `src` que nous développerons les briques élémentaires et que nous les connecterons ensembles afin de former les différents pipelines.
# 
# Avant de rentrer dans les concepts de Kedro, il est **fortement conseillé** (si ce n'est indispensable) de créer un environnement virtuel à la racine du projet.
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# ## Concepts de Kedro
# 
# Kedro ne se contente pas uniquement de créer un projet : il va également apporter des fonctionnalités très puissantes qui en font sa popularité.
# 
# ### Data Catalog
# 
# Pour utiliser des données tout au long d'un pipeline ML, il est déconseillé d'inscrire des chemins d'accès en dur dans le code. Il est préférable d'attribuer des noms à des données existantes que l'on utilise ou celles que l'on crée. C'est rôle du Data Catalog : nous allons définir en amont des référentiels de données avec des noms associés. Le Data Catalog est référencé dans le fichier `conf/base/catalog.yml`.
# 
# Par défaut, Kedro propose plusieurs sous-dossiers dans `data` qui permet de mieux organiser les données.
# 
# - `raw`, `intermediate` et `primary` font référence aux données brutes, celles ayant subi des traitements intermédiaires et celles prêtes à être encodées.
# - `feature` contiendrait la base d'apprentissage $(X,y)$ encodée.
# - `model_input` contiendrait les échantillons d'entraînement et de test fournis au modèle.
# - `models` contiendrait le ou les modèles sérialisés.
# - `model_output` et `reporting` contiendraient les sorties du modèles ainsi que les graphes pour valider et interpéter.
# 
# <div class="alert alert-block alert-info">
#     ℹ️ Bien entendu, nous ne sommes pas tenu de suivre exactement cette structure, il s'agit plutôt d'une organisation par défaut proposée par Kedro.
# </div>
# 
# Dans notre cas, nous n'allons pas effectuer la transformation des données avec Kedro, puisque ce sera une tâche Spark SQL qui en sera chargée. Ainsi, nous aurons uniquement le données transformé qui subira ensuite l'encodage. Nous considérons donc que l'échantillon que recevra Kedro sera situé dans `primary` et sera nommé `primary.csv`.
# 
# À partir de ces données `primary.csv`, nous encoderons vers un nouveau fichier `dataset.csv` dans le dossier `feature`, dont nous récupérerons les sous-ensembles d'apprentissage et de test. Pour référencer tous ces fichiers dans le Data Catalog, nous éditons le fichier `conf/base/catalog.yml` en spécifiant le nom, le type de données et le chemin en relatif par rapport au dossier racine.
# Here you can define all your data sets by using simple YAML syntax.
#
# Documentation for this file format can be found in "The Data Catalog"
# Link: https://kedro.readthedocs.io/en/stable/05_data/01_data_catalog.html

primary:
  type: pandas.CSVDataSet
  filepath: data/03_primary/primary.csv

dataset:
  type: pandas.CSVDataSet
  filepath: data/04_feature/dataset.csv

X_train:
  type: pandas.CSVDataSet
  filepath: data/05_model_input/X_train.csv

y_train:
  type: pandas.CSVDataSet
  filepath: data/05_model_input/y_train.csv

X_test:
  type: pandas.CSVDataSet
  filepath: data/05_model_input/X_test.csv

y_test:
  type: pandas.CSVDataSet
  filepath: data/05_model_input/y_test.csv
# L'avantage du Data Catalog est la flexibilité d'utilisation. Plutôt que de spécifier le chemin d'accès dans chaque fichier Python, nous pouvons simplement inscrire `primary` comme argument, et Kedro va automatiquement charger en mémoire (ici au format CSV avec `pandas`) ce jeu de données. Ainsi, nous pouvons à tout moment modifier la valeur du chemin `filepath` ici sans altérer tous les fichiers Python.
# 
# ### Nodes et Pipelines
# 
# Parmi les concepts les plus importants, nous retrouvons celui des **nodes** et des **pipelines**.
# 
# Un **node** est un élément unitaire qui représente une tâche. Par exemple, nous pouvons imaginer un node pour encoder le jeu de données, un autre pour construire les sous-ensembles d'entraînement et de test, et un dernier pour calibrer un modèle de Machine Learning.
# 
# Un **pipeline**, à l'instar des pipelines de données, est une succession de nodes qui peuvent être assemblés en séquence ou en parallèle.
# 
# Les pipelines sont une partie très importante dans Kedro. Reprenons le pipeline d'expérimentation où l'on entraîne un modèle de Machine Learning.
# 
# <img src="https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/training/ml_engineer_facebook/img/kedro2.png" />
# 
# Ce qui est important à voir, c'est le caractère séquencé entre les tâches. En particulier, **impossible d'entraîner un modèle sans avoir encodé les données**, tout comme il est impossible d'évaluer un modèle si l'on n'en a pas.
# 
# Kedro va nous permettre de créer ces pipelines, garantissant que les **artifacts** (données construites, modèles, etc) vont être disponibles pour les autres nodes du pipeline. C'est un outil essentiel car il va nous assurer que les traitements sont homogènes et que lorsque l'on souhaitera entraîner un nouveau modèle par exemple, les données subiront exactement le même traitement puisqu'elles passeront par le même pipeline. On évite ainsi les risques d'oubli ou d'erreur de cohérence entre deux exécutions successives, chose qui arrive plus souvent que l'on ne l'imagine avec les Jupyter Notebooks.

# ## Premier pipeline
# 
# Nous allons créer ensemble un premier pipeline qui va contenir deux noeuds.
# 
# - Un premier qui va se charger d'encoder le jeu de données `primary`.
# - Un autre qui va séparer la base de données en deux sous-ensembles d'entraînement et d'apprentissage.
# 
# Commençons tout d'abord par télécharger le fichier d'échantillon dans le dossier `data/03_primary`.
cp ~/data/primary.csv ~/purchase-predict/data/03_primary/primary.csv
# Créons un dossier `processing` dans `src/purchase_predict/pipelines`. Nous allons y ajouter deux fichiers Python `nodes.py` et `pipeline.py`.
# 
# - Le fichier `nodes.py` contient les définitions des fonctions qui seront utilisées par les nodes.
# - Le fichier `pipeline.py` permet de construire le pipeline à partir de nodes qui utiliseront les fonctions du fichier `nodes.py`.
# 
# Puisque nous avons deux noeuds, nous devons construire deux fonctions.
# 
# ### Noeud `encode_features`
# 
# La fonction `encode_features` va récupérer `dataset`, qui correspond au fichier CSV ayant subit les transformations (notamment via Spark SQL).
import pandas as pd

from typing import Dict, Any

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Encode features of data file.
    """
    features = dataset.drop(["user_id", "user_session"], axis=1).copy()

    encoders = []
    for label in ["category", "sub_category", "brand"]:
        features[label] = features[label].astype(str)
        features.loc[features[label] == "nan", label] = "unknown"
        encoder = LabelEncoder()
        features.loc[:, label] = encoder.fit_transform(features.loc[:, label].copy())
        encoders.append((label, encoder))

    features["weekday"] = features["weekday"].astype(int)
    return dict(features=features, transform_pipeline=encoders)
# Cette fonction retourne le DataFrame `features`, qui correspond aux données encodées.
# 
# ### Noeud `split_dataset`
# 
# L'autre fonction, `split_dataset`, opère simplement une séparation en deux sous-échantillons. L'argument `test_ratio` permettra de spécifier la proportion d'observation à considérer dans le sous-échantillon de test.
def split_dataset(dataset: pd.DataFrame, test_ratio: float) -> Dict[str, Any]:
    """
    Splits dataset into a training set and a test set.
    """
    X = dataset.drop("purchased", axis=1)
    y = dataset["purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=40
    )

    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
# La fonction retourne les quatre DataFrames.
# 
# ### Construction du pipeline
# 
# Pour entamer la construction du pipeline, nous allons tout d'abord définir un **paramètre** Kedro, le `test_ratio`. En effet, il s'agit d'un paramètre de configuration qui doit être initialisé au préalable, et plutôt que d'inscrire en dur dans le code la valeur du ratio pour l'ensemble de test, tout comme le Data Catalog, le fichier `parameters.yml` dans le dossier `conf/base` permet de centraliser tous les paramètres du modèle, Cloud, de traitement de données, etc.
test_ratio: 0.3
# À partir de là, nous pouvons construire notre pipeline. Pour cela, nous utilisons l'objet `Pipeline` de Kedro, qui s'attends à une liste de `node`. Chaque instance de `node` attends trois paramètres.
# 
# - Le nom de la fonction Python qui sera appelée.
# - Les arguments de la fonction (sous forme de liste ou dictionnaire).
# - Les sorties du modèles (sous forme de liste ou dictionnaire).
from kedro.pipeline import Pipeline, node

from .nodes import encode_features, split_dataset

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                encode_features,
                "primary",
                dict(features="dataset", transform_pipeline="transform_pipeline")
            ),
            node(
                split_dataset,
                ["dataset", "params:test_ratio"],
                dict(
                    X_train="X_train",
                    y_train="y_train",
                    X_test="X_test",
                    y_test="y_test"
                )
            )
        ]
    )
# Le premier noeud appelle la fonction `encode_features` avec pour argument le jeu de données `primary`, et le résultat (un seul) retourné par la fonction sera stocké dans le jeu de données `dataset`.
# 
# Le deuxième noeud nécessite le jeu de données `dataset` ainsi que le paramètre `test_ratio`, et retourne les 4 DataFrames qui correspond aux sous-ensembles.
# 
# Notre pipeline est donc en place, et nous pouvons la visualiser ci-dessous.
# 
# <img src="https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/training/ml_engineer_facebook/img/kedro1.jpg" width="800" />
# 
# Avant de lancer le pipeline, installons les quelques dépendances nécessaires dans l'environnement virtuel (qui est vierge par défaut).
pip install pandas scikit-learn
# Pour terminer, il faut créer une instance du pipeline pour pouvoir l'exécuter. Toutes les instances sont définies dans le fichier `hooks.py` à la racine de `src/purchase_predict`. Ajoutons l'importation suivante.
from purchase_predict.pipelines.processing import pipeline as processing_pipeline
# Nous importons donc le fichier `pipeline` dans `pipelines.processing` dont nous créons l'alias `processing_pipeline`. En appelant la fonction `processing_pipeline.create_pipeline()`, cela va instancier un nouveau pipeline qui contient les deux noeuds.
# 
# Ces instanciations doivent être définies dans la fonction `register_pipelines`, qui retourne un dictionnaire où chaque clé est le nom du pipeline et la valeur l'objet pipeline associé.
@hook_impl
def register_pipelines(self) -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    p_processing = processing_pipeline.create_pipeline()

    return {"processing": p_processing}
# C'est notamment ici qu'il sera également possible d'imbriquer séquentiellement plusieurs pipelines entre-eux.
# 
# Notre pipeline est donc prêt, nous pouvons l'exécuter.
kedro run --pipeline processing2021-01-04 17:45:19,425 - kedro.framework.session.session - WARNING - Unable to git describe /home/jovyan/purchase_predict
2021-01-04 17:45:19,463 - root - INFO - ** Kedro project purchase_predict
2021-01-04 17:45:19,486 - kedro.io.data_catalog - INFO - Loading data from `primary` (CSVDataSet)...
2021-01-04 17:45:19,565 - kedro.pipeline.node - INFO - Running node: encode_features([primary]) -> [dataset]
2021-01-04 17:45:19,597 - kedro.io.data_catalog - INFO - Saving data to `dataset` (CSVDataSet)...
2021-01-04 17:45:19,760 - kedro.runner.sequential_runner - INFO - Completed 1 out of 2 tasks
2021-01-04 17:45:19,760 - kedro.io.data_catalog - INFO - Loading data from `dataset` (CSVDataSet)...
2021-01-04 17:45:19,817 - kedro.io.data_catalog - INFO - Loading data from `params:test_ratio` (MemoryDataSet)...
2021-01-04 17:45:19,817 - kedro.pipeline.node - INFO - Running node: split_dataset([dataset,params:test_ratio]) -> [X_test,X_train,y_test,y_train]
2021-01-04 17:45:19,828 - kedro.io.data_catalog - INFO - Saving data to `X_train` (CSVDataSet)...
2021-01-04 17:45:19,918 - kedro.io.data_catalog - INFO - Saving data to `y_train` (CSVDataSet)...
2021-01-04 17:45:19,950 - kedro.io.data_catalog - INFO - Saving data to `X_test` (CSVDataSet)...
2021-01-04 17:45:19,986 - kedro.io.data_catalog - INFO - Saving data to `y_test` (CSVDataSet)...
2021-01-04 17:45:19,999 - kedro.runner.sequential_runner - INFO - Completed 2 out of 2 tasks
2021-01-04 17:45:19,999 - kedro.runner.sequential_runner - INFO - Pipeline execution completed successfully.
2021-01-04 17:45:19,999 - kedro.framework.session.store - INFO - `save()` not implemented for `BaseSessionStore`. Skipping the step
# En déroulant le dossier `data/05_model_input`, nous devrions voir apparaître les 4 fichiers CSV générés par le pipeline.
# 
# ### Visualisation des pipelines
# 
# La dernière dépendance `kedro-viz` peut être utile car elle permet de visualiser les différents pipelines directement depuis le navigateur.
pip install kedro-viz
# Pour lancer le serveur de visualisation, il suffit d'exécuter la commande suivante (attention de vérifier que l'environnement virtuel est bien activé).
kedro viz --port 4141
# Une fois lancé, nous pouvons y accéder <a href="jupyter://user-redirect/proxy/4141/" target="_blank">par ce lien</a>.

# ## ✔️ Conclusion
# 
# Tu viens de mettre en place ton premier pipeline avec Kedro !
# 
# - Nous avons vu créé notre premier projet avec Kedro.
# - Nous avons détaillé les concepts importants que l'on rencontre avec Kedro.
# - Nous avons mis en place un premier pipeline.
# 
# > ➡️ Après avoir construit le pipeline de traitement de données, place au **pipeline d'entraînement du modèle**.
