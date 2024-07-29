from flask import Flask, request, jsonify
from joblib import load
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
model = load('model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Initialize stopwords, stemmer, and punctuation list
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
punctuation = set(string.punctuation)

def preprocess_text(text):
    words = word_tokenize(text.lower())  # Convert to lowercase
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words and word not in punctuation]
    return ' '.join(cleaned_words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    preprocessed_text = preprocess_text(text)
    # Transform the text to the TF-IDF feature space
    tfidf_features = vectorizer.transform([preprocessed_text])
    # Predict using the model
    prediction = model.predict(tfidf_features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
