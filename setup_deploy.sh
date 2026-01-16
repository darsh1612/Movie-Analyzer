#!/bin/bash

echo "🚀 Setting up Movie Sentiment Analyzer for deployment..."
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
fi

# Install Git LFS if not installed
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS not found. Installing..."
    echo "Please run: brew install git-lfs"
    echo "Then rerun this script."
    exit 1
fi

# Set up Git LFS
echo "📦 Setting up Git LFS for large model files..."
git lfs install
git lfs track "*.pkl"

# Add all files
echo "📝 Adding files to git..."
git add .

# Commit
echo "💾 Committing files..."
git commit -m "Initial commit: Movie Sentiment Analyzer with ML models"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a GitHub repository at https://github.com/new"
echo "2. Run: git remote add origin https://github.com/YOUR-USERNAME/REPO-NAME.git"
echo "3. Run: git push -u origin main"
echo "4. Deploy on Streamlit Cloud: https://share.streamlit.io"
echo ""
echo "See DEPLOYMENT.md for detailed instructions!"
