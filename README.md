# Classification d'images naturelles : comparaison de trois modèles

Application Streamlit qui permet de tester et de comparer trois modèles de classification d'images entraînés sur le jeu de données Natural Images (8 classes : avion, voiture, chat, chien, fleur, fruit, moto, personne).

Application en ligne : https://cnnappmobilnetinterface-pjrucvrfgwjb6gelwavar9.streamlit.app/

## Modèles

| Modèle | Description |
|---|---|
| CNN base | Trois blocs de convolution, GlobalAveragePooling et deux couches Dense |
| CNN amélioré | Quatre blocs de convolution, augmentation de données et dropout |
| MobileNetV2 (TL) | Transfert d'apprentissage depuis ImageNet, base gelée puis fine-tuning des 30 dernières couches |

## Fonctionnalités

- **Prédiction** : envoi d'une image (JPG, PNG, WEBP, BMP, GIF, TIFF, et HEIC si pillow-heif est installé) et affichage de la classe prédite, de la confiance, du top 3 et du temps de prédiction pour chaque modèle.
- **Comparaison** : tableau des indicateurs, courbes de validation et histogrammes des trois modèles.
- **Évaluation** : courbes d'entraînement, matrice de confusion et résultats par classe pour le modèle choisi.

## Structure du projet

```
.
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── models/
    ├── cnn_base.keras
    ├── cnn_ameliore.keras
    ├── mobilenetv2_tl.keras
    └── info.json
```

Le fichier `info.json` contient les noms des classes, le nombre de paramètres, le temps d'entraînement, l'historique d'entraînement et les résultats sur le jeu de test.

## Installation en local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Sur Mac ou Linux, l'activation se fait avec `source .venv/bin/activate`.

## Prétraitement

Les images sont converties en RGB, redimensionnées en 128 x 128 pixels et normalisées entre 0 et 1, comme pendant l'entraînement.

## Déploiement

L'application est déployée sur Streamlit Community Cloud à partir de ce dépôt, avec Python 3.12 et les versions fixées dans `requirements.txt`.

## Auteur

HAFSSA MIFTAH IDRISSI, Master SDIA, ENSET Mohammedia
