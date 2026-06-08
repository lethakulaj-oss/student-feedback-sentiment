import numpy as np
import pickle
import random
import os
import nltk

# Download NLTK data at startup (needed for cloud deployment)
nltk.download('stopwords', quiet=True)

os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tf_keras.models import load_model
from tf_keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import string

import os

# Get the absolute path of the project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Define correct paths
bilstm_model_path = os.path.join(BASE_DIR, '../model/bilstm_model.h5')
cnn_model_path = os.path.join(BASE_DIR, '../model/cnn_model.h5')
cnn_lstm_model_path = os.path.join(BASE_DIR, '../model/cnn_lstm_model.h5')

# Load models
bilstm_model = load_model(bilstm_model_path)
cnn_model = load_model(cnn_model_path)
cnn_lstm_model = load_model(cnn_lstm_model_path)

# Load Tokenizer and Label Encoder
tokenizer_path = os.path.join(BASE_DIR, '../model/tokenizer.pkl')
label_encoder_path = os.path.join(BASE_DIR, '../model/label_encoder.pkl')

import sys
import tf_keras.preprocessing.text as _kpt
# Patch old keras module paths so legacy pickled tokenizers load correctly
import types
_keras_compat = types.ModuleType("keras")
_keras_preprocessing = types.ModuleType("keras.preprocessing")
_keras_preprocessing_text = types.ModuleType("keras.preprocessing.text")
_keras_preprocessing_text.Tokenizer = _kpt.Tokenizer
_keras_compat.preprocessing = _keras_preprocessing
_keras_preprocessing.text = _keras_preprocessing_text
sys.modules.setdefault("keras", _keras_compat)
sys.modules["keras.preprocessing"] = _keras_preprocessing
sys.modules["keras.preprocessing.text"] = _keras_preprocessing_text

import pickle
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)


# Emotion Mapping and Recommendations
label_mapping = {
    "neutral": "Focused",
    "joy": "Cheerful",
    "sadness": "Struggling",
    "love": "Appreciative",
    "anger": "Frustrated",
    "fear": "Anxious",
    "happiness": "Confident",
    "surprise": "Surprised",
    "relief": "Relieved",
    "hate": "Disengaged",
    "fun": "Playful",
    "enthusiasm": "Motivated",
    "empty": "Disconnected",
    "worry": "Worried",
    "boredom": "Bored"
}

recommendations = {
    "Focused": ["Maintain the current teaching pace.", "Ask a probing question.", "Use a real-life example."],
    "Cheerful": ["Acknowledge positive comments.", "Use fun activities.", "Encourage participation."],
    "Struggling": ["Slow down pace.", "Provide visual aids.", "Encourage questions."],
    "Appreciative": ["Thank students.", "Encourage them to share thoughts.", "Build rapport."],
    "Frustrated": ["Address calmly.", "Use simpler explanations.", "Pause to clarify."],
    "Anxious": ["Reassure students.", "Break down complex topics.", "Use calming language."],
    "Confident": ["Challenge with advanced questions.", "Acknowledge their confidence.", "Involve them in discussions."],
    "Surprised": ["Explain unexpected concepts.", "Encourage discussion.", "Use real-life examples."],
    "Relieved": ["Provide summary.", "Share similar examples.", "Encourage reflective thinking."],
    "Disengaged": ["Use interactive elements.", "Provide relatable examples.", "Break the lecture into chunks."],
    "Playful": ["Introduce gamification.", "Use humor.", "Encourage creativity."],
    "Motivated": ["Encourage leadership roles.", "Recognize participation.", "Assign personalized tasks."],
    "Disconnected": ["Simplify content.", "Use quizzes.", "Reach out privately."],
    "Worried": ["Break down topics.", "Provide additional resources.", "Use stress reduction techniques."],
    "Bored": ["Change activity.", "Use storytelling.", "Introduce interactive exercises."]
}

def preprocess_input_text(text):
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(rf"[{string.punctuation}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)
    return text

def predict_with_recommendations(text):
    preprocessed_text = preprocess_input_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    models = {
        "BiLSTM": bilstm_model,
        "CNN": cnn_model,
        "CNN-LSTM": cnn_lstm_model
    }

    results = {}
    for model_name, model in models.items():
        pred_prob = model.predict(padded_sequence)
        pred_class = np.argmax(pred_prob, axis=1)
        original_label = label_encoder.inverse_transform(pred_class)[0]
        updated_label = label_mapping.get(original_label, original_label)
        recommendation = random.choice(recommendations.get(updated_label, ["No recommendation available"]))
        
        results[model_name] = {
            "Emotion": updated_label,
            "Recommendation": recommendation
        }

    return results
