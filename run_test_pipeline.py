from mongo_utils import get_mongo_collection
from transformers import pipeline
from datetime import datetime

# Connexion Mongo
collection = get_mongo_collection("emotions_db", "goemotions_samples")

# Modèle émotionnel
classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      top_k=1)

# Messages de test
messages = [
    "I'm so proud of our team!",
    "This project is exhausting...",
    "I feel totally ignored.",
    "What a wonderful opportunity!"
]

# Analyse + insertion
for msg in messages:
    results = classifier(msg)  # <= ici on reçoit une LISTE
    result = results[0]        # <= on prend le premier dict de la liste
    doc = {
        "texte": msg,
       "emotion": result[0]['label'],
        "score": result['score'],
        "date_insertion": datetime.utcnow(),
    }
    collection.insert_one(doc)
    print(f"> {msg} → {result['label']} ({round(result['score']*100, 2)}%)")

print("🎉 Test complet terminé ! Tous les documents sont enregistrés.")
