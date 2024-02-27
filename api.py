from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer


# Charger le modèle
text_classifier = load('text_classifier.joblib')
tfIdfVectorizer = load('tfIdfVectorizer.joblib')

def is_toxic(text):
    # Prédire la toxicité
    return text_classifier.predict(tfIdfVectorizer.transform([text]))[0][0] == 1


print(is_toxic("I love you"))





