# 🤖 Knowledge AI Agent

A powerful document-based question-answering system that uses Google's Gemini AI to provide intelligent responses based on your uploaded documents.

## 📖 Overview

The Knowledge AI Agent is a Streamlit web application that allows users to upload various document formats (PDF, TXT, DOCX, DOC) and ask questions about their content. The system uses semantic search to find relevant information and generates accurate, context-aware answers using Google's Gemini 2.0 Flash model.

## ✨ Features

### 📚 Document Processing
- **Multi-format Support**: Upload PDF, TXT, DOCX, and DOC files
- **Intelligent Chunking**: Automatic text segmentation with overlap for context preservation
- **TF-IDF Vectorization**: Efficient document indexing and retrieval

### 🔍 Smart Search & QA
- **Semantic Search**: Find relevant document chunks using cosine similarity
- **Context-Aware Answers**: Generate responses based solely on document content
- **Source Attribution**: See which documents provided the information
- **Relevance Scoring**: View similarity scores for retrieved chunks

### 🎨 User Experience
- **Streamlit Web Interface**: Clean, intuitive UI
- **Real-time Chat**: Interactive Q&A interface
- **Document Management**: Easy upload and management in sidebar
- **System Status**: Clear visibility of API and document status

## 🛠️ Tech Stack & APIs

### Core Technologies
- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.0 Flash
- **ML Libraries**: Scikit-learn (TF-IDF, Cosine Similarity)
- **Document Processing**: PyPDF2, python-docx
- **Environment Management**: python-dotenv

### APIs Used
- **Google Generative AI API**: For answer generation using Gemini 2.0 Flash
- **No external vector databases**: Uses in-memory TF-IDF for efficiency

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- Google API key for Gemini AI

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shubha-Bhat/AI-Agent.git
   cd AI-Agent
