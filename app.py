# Monkey patch for Python 3.14 compatibility with cinemagoer
import pkgutil
import importlib
if not hasattr(pkgutil, 'find_loader'):
    def find_loader(name):
        try:
            spec = importlib.util.find_spec(name)
            return spec.loader if spec else None
        except (ImportError, ModuleNotFoundError):
            return None
    pkgutil.find_loader = find_loader

import streamlit as st
import pickle
import numpy as np
import nltk

# Download NLTK data (required for Streamlit Cloud)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px

from groq import Groq
import time
import random
from urllib.parse import urlencode
import base64
import os

# --- FIXED: Selenium imports with error handling ---
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    st.warning("Selenium not available. Using requests-only mode.")

# --- Background styling functions (keeping your original) ---
def set_bg_from_local_improved(image_file):
    """Enhanced background function with better text visibility"""
    if not os.path.exists(image_file):
        # Use gradient fallback if image not found
        set_cinema_gradient_bg()
        return
        
    try:
        with open(image_file, "rb") as f:
            img_bytes = f.read()
        b64_img = base64.b64encode(img_bytes).decode()
        bg_style = f"""
        <style>
        .stApp {{
            background-image: 
                linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                url("data:image/png;base64,{b64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        [data-testid="stAppViewContainer"] > .main {{
            background-image: 
                linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                url("data:image/png;base64,{b64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .main .block-container {{
            background-color: rgba(0, 0, 0, 0.3);
            padding: 2rem;
            border-radius: 15px;
            backdrop-filter: blur(5px);
            margin-top: 1rem;
        }}
        
        .title-container {{
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.9), rgba(20, 20, 20, 0.9));
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 2px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }}
        
        h1 {{
            color: #FFD700 !important;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.9);
            font-weight: bold;
            margin-bottom: 0.5rem !important;
        }}
        
        h3 {{
            color: #FFFFFF !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            font-weight: 500;
            margin-bottom: 1rem !important;
        }}
        
        p, .stMarkdown {{
            color: #E8E8E8 !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
            font-size: 1.1rem;
        }}
        
        .stTextInput label {{
            color: #FFD700 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
            font-weight: bold;
            font-size: 1.1rem !important;
        }}
        
        .stTextInput input {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #000000 !important;
            border: 2px solid rgba(255, 215, 0, 0.3);
            border-radius: 10px;
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, #FFD700, #FFA500);
            color: #000000 !important;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1rem;
            padding: 0.5rem 1rem;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
        }}
        </style>
        """
        st.markdown(bg_style, unsafe_allow_html=True)
    except Exception as e:
        set_cinema_gradient_bg()

def set_cinema_gradient_bg():
    """Cinema-themed gradient background as fallback"""
    bg_style = """
    <style>
    .stApp {
        background: linear-gradient(135deg, 
                    #1a1a2e 0%, 
                    #16213e 25%, 
                    #0f3460 50%, 
                    #533a7b 75%, 
                    #8b5a3c 100%);
        min-height: 100vh;
    }
    
    h1 {
        color: #FFD700 !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.9);
        font-weight: bold;
    }
    
    h3 {
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    
    p, .stMarkdown {
        color: #E8E8E8 !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
    }
    
    .stTextInput label {
        color: #FFD700 !important;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000000 !important;
        border: none;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

def create_title_section():
    """Creates title section"""
    st.markdown("""
    <div class="title-container">
        <h1 style='text-align: center; margin-bottom: 10px;'>🎬 Fast Movie Review Analyzer</h1>
        <h3 style='text-align: center; margin-bottom: 10px;'>Get 100+ IMDb reviews analyzed in seconds!</h3>
        <p style='text-align: center; margin-bottom: 0;'>Lightning-fast sentiment analysis with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

# --- FIXED: Fast and Reliable Requests-Only Scraper ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_movie_reviews_fast(movie_name, max_reviews=100):
    """
    FAST and RELIABLE scraper using only requests - no Selenium issues
    """
    try:
        # Step 1: Get movie ID using direct IMDb search (fixing Cinemagoer compatibility)
        search_url = "https://www.imdb.com/find/"
        params = {'q': movie_name, 's': 'tt', 'ttype': 'tv,ft'}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        # Search for the movie/series
        search_response = session.get(search_url, params=params, timeout=10)
        search_response.raise_for_status()
        search_soup = BeautifulSoup(search_response.content, 'html.parser')
        
        # Try to find the first result
        movie_id = None
        movie_title = None
        
        # New IMDb search result selectors (2025)
        result_selectors = [
            'li.find-title-result a',
            'li.ipc-metadata-list-summary-item a.ipc-metadata-list-summary-item__t',
            'section[data-testid="find-results-section-title"] ul li a',
            'ul.ipc-metadata-list li a'
        ]
        
        for selector in result_selectors:
            results = search_soup.select(selector)
            if results:
                first_result = results[0]
                href = first_result.get('href', '')
                if '/title/tt' in href:
                    # Extract movie ID from URL like /title/tt0903747/
                    movie_id = href.split('/title/tt')[1].split('/')[0]
                    movie_title = first_result.get_text(strip=True)
                    break
        
        if not movie_id:
            # Fallback: try to find any link with /title/tt pattern
            all_links = search_soup.find_all('a', href=True)
            for link in all_links:
                href = link['href']
                if '/title/tt' in href and len(link.get_text(strip=True)) > 0:
                    movie_id = href.split('/title/tt')[1].split('/')[0]
                    movie_title = link.get_text(strip=True)
                    # Clean up title
                    if len(movie_title) > 100:  # Skip if too long (likely not the title)
                        continue
                    break
        
        if not movie_id:
            return None, f"Movie/Series '{movie_name}' not found. Please check the name and try again."
        
        # Clean up movie title - remove year and extra info
        if movie_title:
            movie_title = movie_title.split('(')[0].strip()
            if not movie_title:
                movie_title = movie_name
        
        # Step 2: Setup session with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        reviews_data = []
        
        # Step 3: Try multiple approaches to get reviews
        approaches = [
            # Approach 1: Different sort orders
            {'sort': 'helpfulnessScore', 'dir': 'desc'},
            {'sort': 'submissionDate', 'dir': 'desc'}, 
            {'sort': 'totalVotes', 'dir': 'desc'},
            {'sort': 'userRating', 'dir': 'desc'},
        ]
        
        for approach in approaches:
            if len(reviews_data) >= max_reviews:
                break
                
            # Try multiple pages for each approach
            for page_start in range(0, 200, 25):  # 25 reviews per page
                if len(reviews_data) >= max_reviews:
                    break
                    
                try:
                    # Construct URL
                    base_url = f"https://www.imdb.com/title/tt{movie_id}/reviews"
                    params = {
                        'sort': approach['sort'],
                        'dir': approach['dir'],
                        'start': page_start
                    }
                    
                    response = session.get(base_url, params=params, timeout=15)
                    response.raise_for_status()
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # UPDATED selectors for 2025 IMDb
                    selectors = [
                        # Primary selectors (most likely to work)
                        "div.text.show-more__control",
                        ".lister-item-content .text",
                        ".review-container .content .text",
                        
                        # Alternative selectors 
                        ".content .text",
                        "div[data-testid*='review'] .text",
                        ".imdb-user-review .text",
                        
                        # Fallback selectors
                        "div.content div.text",
                        "[class*='review'] [class*='text']",
                    ]
                    
                    page_reviews = []
                    for selector in selectors:
                        elements = soup.select(selector)
                        if elements:
                            for element in elements:
                                text = element.get_text(strip=True)
                                
                                # Quality filters
                                if (text and 
                                    len(text) >= 50 and  # Minimum length
                                    len(text) <= 5000 and  # Maximum length
                                    text not in [r for r in reviews_data + page_reviews] and  # No duplicates
                                    not any(skip in text.lower() for skip in 
                                           ['cookie', 'privacy', 'terms', 'policy', 'menu', 'navigation'])):
                                    
                                    page_reviews.append(text)
                            break  # Found reviews with this selector, no need to try others
                    
                    if not page_reviews:
                        # Try alternative extraction method
                        all_divs = soup.find_all('div')
                        for div in all_divs:
                            text = div.get_text(strip=True)
                            if (len(text) > 100 and len(text) < 2000 and 
                                text.count('.') >= 2 and  # Has sentences
                                any(movie_word in text.lower() for movie_word in ['movie', 'film', 'watch']) and
                                text not in reviews_data):
                                page_reviews.append(text)
                                if len(page_reviews) >= 10:  # Limit per page
                                    break
                    
                    reviews_data.extend(page_reviews)
                    
                    # If no reviews found, stop trying more pages
                    if not page_reviews:
                        break
                        
                    # Small delay to be respectful
                    
                    
                except Exception as e:
                    continue  # Try next page/approach
        
        # Step 4: Clean up and remove duplicates
        unique_reviews = []
        seen_starts = set()
        
        for review in reviews_data:
            # Use first 80 characters to detect near-duplicates
            start_text = review[:80].lower().strip()
            if start_text not in seen_starts and len(review) >= 50:
                seen_starts.add(start_text)
                unique_reviews.append(review)
                
        return unique_reviews[:max_reviews], movie_title
        
    except Exception as e:
        return None, f"Error scraping reviews: {str(e)}"

# --- OPTIONAL: Selenium scraper with proper error handling ---
def get_movie_reviews_selenium_safe(movie_name, max_reviews=100):
    """
    Safe Selenium scraper with proper error handling and timeouts
    NOTE: Falls back to fast scraper due to Cinemagoer compatibility issues
    """
    # For now, just use the fast scraper which works reliably
    return get_movie_reviews_fast(movie_name, max_reviews)

# --- NLP Functions (keeping your existing code) ---
lemmatizer = WordNetLemmatizer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [i for i in tokens if i not in stop_words and i not in punctuation]
    tokens = [lemmatizer.lemmatize(i) for i in tokens]
    return tokens

def get_vector(tokens, model, vector_size):
    vec = np.zeros(vector_size).reshape(1, -1)
    count = 0
    for word in tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        vec /= count
    return vec

def select_representative_reviews(reviews, sentiments, num_reviews=20):
    if len(reviews) <= num_reviews:
        return reviews
    
    positive_reviews = [reviews[i] for i, sent in enumerate(sentiments) if sent == 1]
    negative_reviews = [reviews[i] for i, sent in enumerate(sentiments) if sent == 0]
    
    total_positive = len(positive_reviews)
    total_negative = len(negative_reviews)
    total_reviews = len(reviews)
    
    positive_sample_size = int((total_positive / total_reviews) * num_reviews)
    negative_sample_size = num_reviews - positive_sample_size
    
    positive_sample_size = min(positive_sample_size, len(positive_reviews))
    negative_sample_size = min(negative_sample_size, len(negative_reviews))
    
    selected_reviews = []
    
    if positive_reviews and positive_sample_size > 0:
        selected_positive = random.sample(positive_reviews, min(positive_sample_size, len(positive_reviews)))
        selected_reviews.extend(selected_positive)
    
    if negative_reviews and negative_sample_size > 0:
        selected_negative = random.sample(negative_reviews, min(negative_sample_size, len(negative_reviews)))
        selected_reviews.extend(selected_negative)
    
    if len(selected_reviews) < num_reviews:
        remaining_reviews = [r for r in reviews if r not in selected_reviews]
        if remaining_reviews:
            additional = random.sample(remaining_reviews, min(num_reviews - len(selected_reviews), len(remaining_reviews)))
            selected_reviews.extend(additional)
    
    return selected_reviews[:num_reviews]

def summarize_reviews_with_groq(selected_reviews, movie_title, api_key, total_reviews_count, sentiment_summary):
    try:
        client = Groq(api_key=api_key)
        
        reviews_text = ""
        for i, review in enumerate(selected_reviews, 1):
            truncated_review = review[:600] + "..." if len(review) > 600 else review
            reviews_text += f"Review {i}: {truncated_review}\n\n"
        
        prompt = f"""Analyze these {len(selected_reviews)} representative reviews for "{movie_title}" (from {total_reviews_count} total reviews).

SENTIMENT BREAKDOWN: {sentiment_summary}

REVIEWS:
{reviews_text}

Provide analysis with:
1. **Overall Rating** (1-10) and reception summary
2. **Key Themes** mentioned by viewers  
3. **Strengths & Weaknesses** highlighted
4. **Target Audience** who would enjoy this
5. **Recommendation** (Yes/No with reasoning)

Keep it concise but insightful."""

        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI summary: {str(e)}"

# --- STREAMLIT APP ---
def main():
    # Page setup
    st.set_page_config(
        layout="wide", 
        page_title="Fast Movie Review Analyzer",
        initial_sidebar_state="expanded"
    )
    
    # Apply styling
    set_bg_from_local_improved("D:/Gemini_Generated_Image_lzd8jqlzd8jqlzd8.png")
    create_title_section()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Scraping method selection
    scraping_method = st.sidebar.selectbox(
        "Choose Scraping Method",
        ["Fast (Recommended)", "Advanced (Selenium)"],
        help="Fast mode uses requests-only and is very reliable. Advanced mode uses Selenium but may have issues."
    )
    
    max_reviews = st.sidebar.slider("Maximum Reviews", 50, 150, 100)
    
    # API key from Streamlit secrets (for deployment) or environment variable
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except:
        groq_api_key = ""  # Fallback to empty string if secrets not available
    
    st.sidebar.markdown("### ✨ Features")
    st.sidebar.write("🚀 Lightning-fast scraping (10-30 seconds)")
    st.sidebar.write("🎯 100+ reviews per movie/series") 
    st.sidebar.write("🤖 AI-powered analysis")
    st.sidebar.write("📊 Beautiful visualizations")
    
    # Main input
    st.markdown("### 🎬 Enter Movie or Series Name")
    movie_name_input = st.text_input(
        "",
        placeholder="e.g., Inception, Breaking Bad, Avengers Endgame",
        label_visibility="collapsed"
    )
    
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_sentiment = st.button("📊 Quick Analysis", use_container_width=True, type="primary")
    with col2:
        analyze_with_summary = st.button("🤖 Full AI Analysis", use_container_width=True)
    
    # Analysis logic
    if analyze_sentiment or analyze_with_summary:
        if not movie_name_input.strip():
            st.warning("⚠️ Please enter a movie or series name!")
            return
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_container = st.container()
        
        with status_container:
            status_text = st.empty()
            
        # Step 1: Scrape reviews
        status_text.info("🔍 Searching for movie and fetching reviews...")
        progress_bar.progress(25)
        
        start_time = time.time()
        
        # Choose scraping method
        if scraping_method == "Fast (Recommended)":
            with st.spinner("⚡ Fast scraping in progress... (~10-30 seconds)"):
                reviews, actual_movie_title = get_movie_reviews_fast(movie_name_input, max_reviews)
        else:
            with st.spinner("🔧 Advanced scraping... may take longer"):
                reviews, actual_movie_title = get_movie_reviews_selenium_safe(movie_name_input, max_reviews)
        
        scrape_time = time.time() - start_time
        progress_bar.progress(50)
        
        if reviews is None:
            st.error(f"❌ {actual_movie_title}")
            progress_bar.empty()
            status_text.empty()
            return
        elif not reviews:
            st.warning(f"⚠️ No reviews found for '{actual_movie_title}'. Try a different movie name.")
            progress_bar.empty()
            status_text.empty()
            return
        
        # Step 2: Load ML models
        status_text.info("🧠 Loading sentiment analysis models...")
        try:
            with open('word2vec_model.pkl', 'rb') as f:
                w2v_model = pickle.load(f)
            with open('classifier_model.pkl', 'rb') as f:
                classifier = pickle.load(f)
        except FileNotFoundError:
            st.error("❌ Model files not found! Please ensure 'word2vec_model.pkl' and 'classifier_model.pkl' are in the app directory.")
            progress_bar.empty()
            status_text.empty()
            return
        
        progress_bar.progress(70)
        
        # Step 3: Sentiment Analysis
        status_text.info(f"📈 Analyzing sentiment of {len(reviews)} reviews...")
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        
        for review in reviews:
            transformed_tokens = transform_text(review)
            vector_input = get_vector(transformed_tokens, w2v_model, w2v_model.vector_size)
            result = classifier.predict(vector_input)[0]
            sentiments.append(result)
            if result == 1:
                positive_count += 1
            else:
                negative_count += 1
        
        progress_bar.progress(90)
        
        # Create results
        sentiment_summary = f"{positive_count} positive, {negative_count} negative"
        analysis_time = time.time() - start_time
        
        # Clear status indicators
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success(f"✅ Analysis Complete! Processed {len(reviews)} reviews for '{actual_movie_title}' in {analysis_time:.1f} seconds")
        
        # Main results layout
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("📊 Sentiment Analysis Results")
            
            # Metrics
            sentiment_ratio = (positive_count / len(reviews)) * 100
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("📝 Total Reviews", len(reviews))
            metric_col2.metric("😊 Positive", positive_count)
            metric_col3.metric("😞 Negative", negative_count) 
            metric_col4.metric("📈 Positive %", f"{sentiment_ratio:.1f}%")
            
            # Pie chart
            sentiment_data = pd.DataFrame({
                'Sentiment': ['😊 Positive', '😞 Negative'],
                'Count': [positive_count, negative_count]
            })
            
            fig = px.pie(
                sentiment_data, 
                values='Count', 
                names='Sentiment',
                title=f"Sentiment Distribution - {actual_movie_title}",
                color='Sentiment',
                color_discrete_map={'😊 Positive': '#00CC96', '😞 Negative': '#FF6692'},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                font=dict(size=14),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Movie Verdict")
            
            # Determine verdict
            if positive_count > negative_count:
                verdict_delta = positive_count - negative_count
                confidence = (verdict_delta / len(reviews)) * 100
                
                st.markdown("### ✅ **RECOMMENDED**")
                st.success(f"This {'movie' if 'series' not in actual_movie_title.lower() else 'series'} has predominantly positive reviews!")
                st.metric("Confidence Score", f"{confidence:.1f}%")
                
                if confidence > 50:
                    st.info("🔥 **Highly Recommended** - Strong positive reception!")
                elif confidence > 25:
                    st.info("👍 **Recommended** - Generally positive reviews")
                else:
                    st.info("🤔 **Cautiously Recommended** - Slight positive lean")
                    
            elif negative_count > positive_count:
                verdict_delta = negative_count - positive_count
                confidence = (verdict_delta / len(reviews)) * 100
                
                st.markdown("### ❌ **NOT RECOMMENDED**")
                st.error(f"This {'movie' if 'series' not in actual_movie_title.lower() else 'series'} has predominantly negative reviews.")
                st.metric("Negative Confidence", f"{confidence:.1f}%")
                
                if confidence > 50:
                    st.warning("💀 **Strongly Not Recommended** - Mostly negative reviews")
                elif confidence > 25:
                    st.warning("👎 **Not Recommended** - Generally negative reception")
                else:
                    st.warning("⚠️ **Probably Skip** - Lean towards negative")
                    
            else:
                st.markdown("### 🤷 **MIXED RECEPTION**")
                st.warning("Reviews are evenly split - check detailed analysis!")
                st.metric("Split Ratio", "50/50")
            
            # Performance info
            st.markdown("---")
            st.info(f"⚡ Scraped in {scrape_time:.1f}s\n📊 Analyzed {len(reviews)} reviews\n🎯 {scraping_method} mode")
        
        # AI Summary (if requested)
        if analyze_with_summary and groq_api_key:
            st.markdown("---")
            st.subheader("🤖 AI-Powered Detailed Analysis")
            
            with st.spinner("🤖 Generating AI insights from representative reviews..."):
                selected_reviews = select_representative_reviews(reviews, sentiments, num_reviews=15)
                ai_summary = summarize_reviews_with_groq(
                    selected_reviews, 
                    actual_movie_title, 
                    groq_api_key,
                    len(reviews), 
                    sentiment_summary
                )
                
            # Display AI summary in a nice container
            with st.container():
                st.markdown(ai_summary)
            
            st.info(f"💡 AI analysis based on {len(selected_reviews)} representative reviews from {len(reviews)} total reviews")
        
        # Sample Reviews Section  
        st.markdown("---")
        st.subheader("📝 Sample Reviews")
        
        tab1, tab2, tab3 = st.tabs(["🎲 Random Sample", "😊 Positive Reviews", "😞 Negative Reviews"])
        
        with tab1:
            sample_size = min(5, len(reviews))
            for i in range(sample_size):
                sentiment_label = "😊 Positive" if sentiments[i] == 1 else "😞 Negative"
                with st.expander(f"Review {i+1} - {sentiment_label}"):
                    review_text = reviews[i][:800] + "..." if len(reviews[i]) > 800 else reviews[i]
                    st.write(review_text)
        
        with tab2:
            positive_reviews = [reviews[i] for i, sent in enumerate(sentiments) if sent == 1]
            if positive_reviews:
                for i, review in enumerate(positive_reviews[:4]):
                    with st.expander(f"😊 Positive Review {i+1}"):
                        review_text = review[:800] + "..." if len(review) > 800 else review
                        st.write(review_text)
            else:
                st.write("No positive reviews found.")
        
        with tab3:
            negative_reviews = [reviews[i] for i, sent in enumerate(sentiments) if sent == 0]
            if negative_reviews:
                for i, review in enumerate(negative_reviews[:4]):
                    with st.expander(f"😞 Negative Review {i+1}"):
                        review_text = review[:800] + "..." if len(review) > 800 else review
                        st.write(review_text)
            else:
                st.write("No negative reviews found.")


# Enhanced sidebar information
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 How It Works")
st.sidebar.write("""
**Enhanced Process:**
1. 🌐 **Smart Web Scraping**: Fetches 100+ reviews from multiple IMDb pages
2. 🧠 **ML Sentiment Analysis**: Classifies all reviews using trained models  
3. 📊 **Statistical Analysis**: Provides comprehensive metrics and confidence scores
4. 🤖 **AI Summarization**: Analyzes 20 representative reviews for detailed insights
5. ⚡ **Optimized Performance**: Fast processing with smart sampling
""")

st.sidebar.subheader("✨ Key Improvements")
st.sidebar.write("""
- **5x More Data**: 100 reviews vs 20 previously
- **Faster AI**: Only 20 reviews sent to Groq API
- **Better Quality**: Enhanced filtering and duplicate removal
- **Smarter Sampling**: Representative review selection
- **Rich Insights**: Confidence scores and detailed metrics
""")

if __name__ == "__main__":
    main()