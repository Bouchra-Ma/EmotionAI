from mongo_utils import get_mongo_collection
from transformers import pipeline
from datetime import datetime

# Récupérer la collection MongoDB
collection = get_mongo_collection("emotions_db", "analyzed_texts")

# Créer le pipeline de détection d’émotions
classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      top_k=3)

# Demander un texte à l’utilisateur
texte = input("Écris ton message ici (en anglais) : ")

# Lancer l’analyse
results = classifier(texte)

# Préparer le document à insérer
doc = {
    "texte": texte,
    "emotions": results,
    "date": datetime.utcnow()
}

# Insérer dans MongoDB
collection.insert_one(doc)

# Affichage
print("Analyse enregistrée dans la base de données ✅")
for res in results:
    for emotion in res:
        print(f"{emotion['label']} : {emotion['score']:.4f}")
