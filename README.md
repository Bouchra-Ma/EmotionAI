# Projet IA émotionnelle

Ce projet utilise le dataset **GoEmotions** de Google pour analyser les émotions dans des textes via un modèle BERT. Les résultats sont stockés dans une base MongoDB.

## Structure
- `1_load_dataset_to_mongo.py` : Importe le dataset dans MongoDB.
- `2_emotion_predictor.py` : Prédit les émotions d’un texte avec un modèle BERT.
- `mongo_utils.py` : Contient la fonction de connexion MongoDB.
