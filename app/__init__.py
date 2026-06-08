import nltk
import os

# Use bundled NLTK data shipped with the app
nltk.data.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'nltk_data'))

from flask import Flask

app = Flask(__name__)

from app import routes  # Importing routes
