import pandas as pd
import numpy as np
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Changed from Stemmer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec # Using Word2Vec

# --- 1. UPDATED PREPROCESSING FUNCTION (USING LEMMATIZATION) ---
lemmatizer = WordNetLemmatizer()
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)   
    tokens = [i for i in tokens if i.isalnum()]
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [i for i in tokens if i not in stop_words and i not in punctuation]
    # Using lemmatization instead of stemming
    tokens = [lemmatizer.lemmatize(i) for i in tokens]
    return tokens # Return a list of tokens for Word2Vec

# --- 2. VECTORIZATION FUNCTION (AVERAGING WORD2VEC VECTORS) ---
def get_vector(tokens, model, vector_size):
    # Create a zero vector of the appropriate size
    vec = np.zeros(vector_size)
    count = 0
    for word in tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        # Average the vectors
        vec /= count
    return vec

print("Starting model training process...")

# Load dataset
try:
    df = pd.read_csv("C:\\Users\\admin\\Desktop\\GenrativeAi\\NLP\\IMDB Dataset.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found.")
    exit()

# Encode labels
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])
print("Labels encoded.")

# Apply preprocessing
print("Applying text transformation... (This may take a few minutes)")
df['tokens'] = df['review'].apply(transform_text)
print("Text transformation complete.")

# --- 3. TRAIN WORD2VEC MODEL ---
print("Training Word2Vec model...")
# vector_size is the dimensionality of the word vectors.
# window is the max distance between the current and predicted word within a sentence.
# min_count ignores all words with a total frequency lower than this.
word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(df['tokens'], total_examples=word2vec_model.corpus_count, epochs=10)
print("Word2Vec model trained.")

# --- 4. CREATE FEATURE VECTORS FOR THE CLASSIFIER ---
print("Creating feature vectors by averaging Word2Vec vectors...")
vector_size = word2vec_model.vector_size
X = np.array([get_vector(tokens, word2vec_model, vector_size) for tokens in df['tokens']])
y = df['sentiment'].values
print("Feature vectors created.")

# --- 5. TRAIN THE SENTIMENT CLASSIFIER ---
model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
model.fit(X, y)
print("Sentiment classifier training complete.")

# --- 6. SAVE THE MODELS ---
with open('word2vec_model.pkl', 'wb') as f:
    pickle.dump(word2vec_model, f)
with open('classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nSuccess! `word2vec_model.pkl` and `classifier_model.pkl` have been created.")