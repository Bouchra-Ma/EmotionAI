from pymongo import MongoClient
from datasets import load_dataset
from mongo_utils import get_mongo_collection

collection = get_mongo_collection("emotions_db", "goemotions_samples")

# Tu peux maintenant insérer, lire ou mettre à jour les documents ici


# Connexion Mongo
client = MongoClient("mongodb://localhost:27017/")
db = client["emotions_db"]
collection = db["goemotions_samples"]

# Charger dataset GoEmotions
dataset = load_dataset("go_emotions", split="train")

# Insertion des 100 premiers exemples dans Mongo
for i in range(100):
    item = dataset[i]
    doc = {
        "text": item["text"],
        "labels": item["labels"]  # Ce sont des indices (on verra comment les décoder)
    }
    collection.insert_one(doc)

print("✅ 100 exemples GoEmotions insérés avec succès !")
