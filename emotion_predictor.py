from transformers import pipeline
from mongo_utils import get_mongo_collection
from datetime import datetime
from mongo_utils import get_mongo_collection

collection = get_mongo_collection("emotions_db", "goemotions_samples")



# Accès à la collection des résultats
collection = get_mongo_collection("emotions_db", "emotion_analysis_results")

# Chargement du modèle BERT
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotions", top_k=None)

# Texte à analyser
texte = "I'm very excited to start this project, but also a bit nervous."

# Prédiction des émotions
resultats = classifier(texte)
emotions = [res["label"] for res in resultats[0] if res["score"] > 0.3]

# Insertion dans Mongo
doc = {
    "texte": texte,
    "emotions_detectees": emotions,
    "timestamp": datetime.now()
}
collection.insert_one(doc)

print("✅ Analyse enregistrée :", doc)

