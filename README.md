# AI Resume Screener

AI-powered resume screening and ranking system built with NLP, sentence embeddings and Streamlit.

**Live demo:** https://resume-screener-aum5c6c7qn6wm7pemv9waa.streamlit.app

## What it does
Upload multiple PDF resumes and a job description. The app ranks candidates by match score, shows matched and missing skills, and generates a hiring recommendation.

## How it works
1. Extracts text from PDFs (pdfplumber)
2. Preprocesses text with NLTK (tokenization, stopwords, lemmatization)
3. Generates semantic embeddings with all-MiniLM-L6-v2 (sentence-transformers)
4. Ranks candidates using cosine similarity
5. Compares against a TF-IDF baseline
6. Extracts skills and measures coverage of the job requirements
7. Final score = weighted mix of semantic similarity and skill coverage (adjustable)
8. Generates AI feedback with the Groq API (optional), or rule-based feedback as a fallback

## Tech stack
Python, Streamlit, sentence-transformers, scikit-learn, pandas, NumPy, NLTK, pdfplumber, Groq API

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Limitations
- Only reads PDFs with selectable text (no OCR)
- Fixed English skill list, editable in utils.py
- AI models can be biased: this tool supports recruiters and does not replace human decisions

## Future improvements
- OCR for scanned PDFs
- Skill extraction with spaCy
- Vector search with FAISS
- Evaluation against rankings made by human recruiters
