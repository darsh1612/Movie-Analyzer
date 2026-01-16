# 🚀 Deployment Guide: Streamlit Community Cloud

## Why Streamlit Cloud? 

✅ **Free forever**  
✅ **Built for Streamlit apps**  
✅ **Easy GitHub integration**  
✅ **No configuration needed**  
✅ **Auto-updates from GitHub**

---

## 📋 Prerequisites

Before deploying, make sure you have:

1. ✅ A **GitHub account** (create one at [github.com](https://github.com))
2. ✅ **Groq API key** (get free at [console.groq.com](https://console.groq.com))
3. ✅ All required files in your project (created ✓)

---

## 🎯 Step-by-Step Deployment

### **Step 1: Create a GitHub Repository**

1. Go to [github.com](https://github.com) and sign in
2. Click the **"+" icon** (top right) → **"New repository"**
3. Repository settings:
   - **Name**: `movie-sentiment-analyzer` (or any name you like)
   - **Visibility**: Public (required for free Streamlit Cloud)
   - **Initialize**: DON'T check "Add README" (we already have one)
4. Click **"Create repository"**

---

### **Step 2: Push Your Code to GitHub**

Open Terminal in your project directory and run:

```bash
cd "/Users/darshgupta/Downloads/NLP 2"

# Initialize git repository
git init

# Add all files
git add .

# Commit your code
git commit -m "Initial commit: Movie Sentiment Analyzer"

# Add GitHub remote (replace YOUR-USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/movie-sentiment-analyzer.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note**: Replace `YOUR-USERNAME` with your actual GitHub username!

If asked for credentials:
- Username: Your GitHub username
- Password: Use a **Personal Access Token** (not your password)
  - Get token at: [github.com/settings/tokens](https://github.com/settings/tokens)
  - Generate new token (classic) → Select "repo" scope → Copy the token

---

### **Step 3: Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign up with GitHub"** or **"Sign in"**
3. Authorize Streamlit to access your GitHub
4. Click **"New app"**
5. Fill in the details:
   - **Repository**: Select `YOUR-USERNAME/movie-sentiment-analyzer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
6. Click **"Advanced settings"**
7. Set **Python version**: `3.11` (important!)
8. Click **"Save"**

---

### **Step 4: Add Your Groq API Key**

1. Before clicking "Deploy", scroll down to **"Secrets"**
2. Click on **"Secrets"** to expand
3. Add this in the secrets box:
   ```toml
   GROQ_API_KEY = ""
   ```
   *(Replace with your actual Groq API key if different)*

4. Click **"Deploy!"**

---

### **Step 5: Wait for Deployment** 

⏱️ **First deployment takes 5-10 minutes** (installing dependencies)

You'll see logs showing:
- ✅ Installing Python packages
- ✅ Downloading NLTK data
- ✅ Loading your models
- ✅ Starting the app

---

## 🎉 Done! Your App is Live!

Once deployed, you'll get a URL like:
```
https://YOUR-USERNAME-movie-sentiment-analyzer.streamlit.app
```

**Share this URL** on your resume, LinkedIn, or portfolio!

---

## 🔄 Updating Your App

Whenever you make changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will **automatically redeploy** your app! 🚀

---

## ⚠️ Troubleshooting

### **Error: "ModuleNotFoundError: No module named 'gensim'"**
**Solution**: Make sure `gensim` is in `requirements.txt`

### **Error: "NLTK data not found"**
**Solution**: Add this to your `app.py` at the top:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### **Error: "File not found: word2vec_model.pkl"**
**Solution**: Make sure `.pkl` files are committed to GitHub:
```bash
git add *.pkl
git commit -m "Add model files"
git push
```

### **App is slow on first load**
This is normal! The large `.pkl` files (74MB for Word2Vec) take time to load initially. Once loaded, it's fast.

---

## 📊 Alternative Hosting Options

If you need more resources or private repos:

### **Render** (Free tier available)
1. Go to [render.com](https://render.com)
2. Connect GitHub
3. Create new "Web Service"
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run app.py --server.port=$PORT`

### **Railway** (Free tier available)
1. Go to [railway.app](https://railway.app)
2. Connect GitHub
3. Deploy from repo
4. Add environment variable: `GROQ_API_KEY`

---

## 🎓 Tips for Your Portfolio

1. **Add screenshots** to your README
2. **Write a good description** of what it does
3. **Include metrics**: "Analyzes 100+ reviews in 30 seconds"
4. **Add technical details**: Word2Vec + Logistic Regression
5. **Link to live demo** on your resume/LinkedIn

---

## ✅ Checklist

- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Streamlit Cloud account created
- [ ] App deployed
- [ ] Groq API key added to secrets
- [ ] App URL works and tested
- [ ] Link added to resume/portfolio

---

**Need help?** Check the [Streamlit Community Forum](https://discuss.streamlit.io/)

Good luck with your deployment! 🚀
