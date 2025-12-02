#!/bin/bash
# Script to convert Python script to Jupyter Notebook using Jupytext

echo "================================================"
echo "Converting to Jupyter Notebook"
echo "================================================"
echo ""

# Check if jupytext is installed
if ! command -v jupytext &> /dev/null; then
    echo "❌ Jupytext not found!"
    echo ""
    echo "Installing jupytext..."
    pip install jupytext
    echo ""
fi

# Convert to notebook
echo "🔄 Converting train_emotion_recognition.py to .ipynb..."
jupytext --to ipynb train_emotion_recognition.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Conversion successful!"
    echo ""
    echo "================================================"
    echo "Created: train_emotion_recognition.ipynb"
    echo "================================================"
    echo ""
    echo "📤 Upload to Google Colab:"
    echo "   1. Go to https://colab.research.google.com/"
    echo "   2. File > Upload notebook"
    echo "   3. Select train_emotion_recognition.ipynb"
    echo "   4. Runtime > Change runtime type > GPU"
    echo ""
    echo "🚀 Or run locally:"
    echo "   jupyter notebook train_emotion_recognition.ipynb"
    echo ""
    echo "================================================"
else
    echo ""
    echo "❌ Conversion failed!"
    echo "Please check that jupytext is installed correctly."
    echo ""
fi

