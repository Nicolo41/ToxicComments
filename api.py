from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Charger le modèle
tokenizer = load('./joblib/tokenizer.joblib')
model = load('./joblib/GRU.joblib')


# Fonction de préparation du texte : tokenization, padding et troncature
def prepare_text(text):
    trunc_type='post'      # Truncates the tweet if it is longer than max_length
    padding_type='post'    # Adds padding to the end of the tweet if it is shorter than max_length
    max_length = 50   # Maximum size of a tweet

    # Tokenization
    sequences = tokenizer.texts_to_sequences([text])

    # Padding & Truncature
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    return padded_sequences


def predictions(text):
    text = prepare_text(text)
    predict = model.predict(text).round()
    return predict

def is_toxic(text):
    text = prepare_text(text)
    predict = model.predict(text)[0][0].round()
    return predict == 1

def is_severe_toxic(text):
    text = prepare_text(text)
    predict = model.predict(text)[0][1].round()
    return predict == 1

def is_obscene(text):
    text = prepare_text(text)
    predict = model.predict(text)[0][2].round()
    return predict == 1

def is_threat(text):
    text = prepare_text(text)
    predict = model.predict(text)[0][3].round()
    return predict == 1

def is_insult(text):
    text = prepare_text(text)
    predict = model.predict(text)[0][4].round()
    return predict == 1

def is_identity_hate(text):
    text = prepare_text(text)
    predict = model.predict(text)[0][5].round()
    return predict == 1






