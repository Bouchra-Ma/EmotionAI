from transformers import pipeline
from datetime import datetime
from pymongo import MongoClient
from collections import Counter


# Initialisation du modèle
classifier = pipeline(
    "text-classification",
    model="joeddav/distilbert-base-uncased-go-emotions-student",
    return_all_scores=False
)

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["emotions_db"]
collection = db["goemotions_case_study"]

# Messages simulés
messages = [
    "I'm completely lost with this new framework.",
    "Yes!! We finally fixed that weird bug!",
    "Honestly, I'm tired of always having to catch up for others.",
    "I'm so happy I finished my task on time.",
    "This is getting really stressful, I can't take much more.",
    "Great team work today!",
    "This instruction is really unclear, I don't get it.",
    "I feel overwhelmed these days...",
    "Thanks everyone for your help!",
    "We'll never meet the deadlines like this..."
]

# Analyse et insertion
for message in messages:
    result = classifier(message)
    emotion = result[0]['label']
    confidence = round(result[0]['score'], 2)

    collection.insert_one({
        "message": message,
        "emotion": emotion,
        "confidence": confidence,
        "date": datetime.now()
    })

print("✅ Étude de cas enregistrée dans MongoDB !")

# Résumé statistique des émotions
emotions = [doc["emotion"] for doc in collection.find()]
counter = Counter(emotions)

print("\n📊 Résumé des émotions détectées :")
for emotion, count in counter.items():
    print(f"{emotion} : {count}")
