# AI Resume Screener

Ranks PDF resumes against a job description using NLP, sentence embeddings and generative AI.

## How it works
1. Extracts text from PDFs (`pdfplumber`)
2. Preprocesses text (NLTK: tokenization, stopwords, lemmatization)
3. Generates embeddings with `all-MiniLM-L6-v2` (sentence-transformers)
4. Computes cosine similarity between each resume and the job description
5. Compares against a TF-IDF baseline
6. Extracts skills and computes coverage of the job's required skills
7. Final score = weight x semantic similarity + (1 - weight) x skill coverage
8. Generates feedback with AI (Groq) or with simple rules if no key is provided

## How to run
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- Only reads PDFs with selectable text (no OCR for scanned PDFs)
- Fixed skill list (editable in `utils.py`)
- AI models can be biased: the tool supports, but does not replace, human decisions

## Future improvements
- OCR for scanned PDFs
- Skill extraction with spaCy NER
- Evaluation against rankings made by human recruiters