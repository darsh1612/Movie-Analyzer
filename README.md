# 🎬 AI Movie Review Analyzer

AI-powered sentiment analysis web application that scrapes IMDb reviews and provides real-time sentiment classification with confidence scores.

## Features

- 🚀 **Real-time web scraping**: Fetches 100+ IMDb reviews in seconds
- 🧠 **ML-powered sentiment analysis**: Uses Word2Vec embeddings + Scikit-learn classifier
- 🤖 **AI-generated insights**: Groq LLM integration for detailed review summaries
- 📊 **Interactive visualizations**: Plotly charts for sentiment distribution
- ⚡ **Fast & reliable**: Optimized scraping with multiple fallback strategies

## Tech Stack

- **Frontend**: Streamlit
- **NLP**: NLTK, Gensim (Word2Vec)
- **ML**: Scikit-learn (Logistic Regression)
- **AI**: Groq LLM API
- **Web Scraping**: BeautifulSoup, Requests
- **Visualization**: Plotly

## Local Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
   ```
4. Create `.streamlit/secrets.toml` with your Groq API key:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## How It Works

1. **Search**: Enter a movie/series name
2. **Scrape**: App searches IMDb and scrapes 100+ reviews
3. **Analyze**: Each review is processed through:
   - Text preprocessing (tokenization, lemmatization)
   - Word2Vec vectorization
   - Logistic Regression classification
4. **Visualize**: Results shown with sentiment breakdown and confidence scores
5. **AI Summary**: Optional Groq LLM analysis for detailed insights

## Project Structure

```
├── app.py                    # Main Streamlit application
├── train.py                  # Model training script
├── word2vec_model.pkl        # Trained Word2Vec model
├── classifier_model.pkl      # Trained sentiment classifier
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies (for deployment)
└── .streamlit/
    └── secrets.toml.example # API key template
```

## License

MIT License

## Author

Built with ❤️ for NLP learning and portfolio demonstration
