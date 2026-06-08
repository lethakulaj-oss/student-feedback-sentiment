import numpy as np
import pickle
import random
import os
import re
import string
import sys
import types

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

# Hardcoded English stopwords — no NLTK download needed
ENGLISH_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn"
}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Lazy-loaded globals
_models = None
_tokenizer = None
_label_encoder = None

def _load_resources():
    global _models, _tokenizer, _label_encoder
    if _models is not None:
        return  # Already loaded

    from tf_keras.models import load_model
    import tf_keras.preprocessing.text as _kpt

    # Patch old keras module paths so legacy pickled tokenizers load correctly
    _keras_compat = types.ModuleType("keras")
    _keras_preprocessing = types.ModuleType("keras.preprocessing")
    _keras_preprocessing_text = types.ModuleType("keras.preprocessing.text")
    _keras_preprocessing_text.Tokenizer = _kpt.Tokenizer
    _keras_compat.preprocessing = _keras_preprocessing
    _keras_preprocessing.text = _keras_preprocessing_text
    sys.modules.setdefault("keras", _keras_compat)
    sys.modules["keras.preprocessing"] = _keras_preprocessing
    sys.modules["keras.preprocessing.text"] = _keras_preprocessing_text

    # Load tokenizer and label encoder
    tokenizer_path = os.path.join(BASE_DIR, '../model/tokenizer.pkl')
    label_encoder_path = os.path.join(BASE_DIR, '../model/label_encoder.pkl')

    with open(tokenizer_path, 'rb') as f:
        _tokenizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        _label_encoder = pickle.load(f)

    # Load models one at a time
    _models = {
        "BiLSTM": load_model(os.path.join(BASE_DIR, '../model/bilstm_model.h5')),
        "CNN": load_model(os.path.join(BASE_DIR, '../model/cnn_model.h5')),
        "CNN-LSTM": load_model(os.path.join(BASE_DIR, '../model/cnn_lstm_model.h5')),
    }


# Emotion Mapping and Recommendations
label_mapping = {
    "neutral": "Focused", "joy": "Cheerful", "sadness": "Struggling",
    "love": "Appreciative", "anger": "Frustrated", "fear": "Anxious",
    "happiness": "Confident", "surprise": "Surprised", "relief": "Relieved",
    "hate": "Disengaged", "fun": "Playful", "enthusiasm": "Motivated",
    "empty": "Disconnected", "worry": "Worried", "boredom": "Bored"
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
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("english")
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(rf"[{string.punctuation}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in ENGLISH_STOPWORDS)
    return text

def predict_with_recommendations(text):
    from tf_keras.preprocessing.sequence import pad_sequences

    # Load models only on first prediction request
    _load_resources()

    preprocessed_text = preprocess_input_text(text)
    sequence = _tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    results = {}
    for model_name, model in _models.items():
        pred_prob = model.predict(padded_sequence)
        pred_class = np.argmax(pred_prob, axis=1)
        original_label = _label_encoder.inverse_transform(pred_class)[0]
        updated_label = label_mapping.get(original_label, original_label)
        recommendation = random.choice(recommendations.get(updated_label, ["No recommendation available"]))
        results[model_name] = {
            "Emotion": updated_label,
            "Recommendation": recommendation
        }

    return results
