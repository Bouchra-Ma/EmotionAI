from transformers import pipeline

# Charger le modèle de classification émotionnelle
classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores=True)

# Exemple de texte
texte = "i'm so gratful."

# Prédiction
resultats = classifier(texte)

# Trier les résultats par score décroissant
resultats_tries = sorted(resultats[0], key=lambda x: x['score'], reverse=True)

# Afficher les 3 émotions les plus probables
print("Top 3 émotions détectées :")
for item in resultats_tries[:3]:
    print(f"{item['label']} : {item['score']:.4f}")
