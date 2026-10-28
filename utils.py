"""Core functions of the AI Resume Screener."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"

# canonical skill -> variations that may appear in the text
SKILLS = {
    "python": ["python"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "keras": ["keras"],
    "transformers": ["transformers", "hugging face", "huggingface"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "generative ai": ["generative ai", "genai", "gen ai"],
    "langchain": ["langchain"],
    "rag": ["rag", "retrieval augmented generation"],
    "streamlit": ["streamlit"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "mongodb": ["mongodb"],
    "spark": ["spark", "pyspark"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel"],
    "statistics": ["statistics", "statistical"],
    "data visualization": ["data visualization", "matplotlib", "seaborn", "plotly"],
    "computer vision": ["computer vision", "opencv"],
    "java": ["java"],
    "javascript": ["javascript"],
    "c++": ["c++"],
    "rest api": ["rest api", "rest apis", "restful"],
    "nltk": ["nltk"],
    "spacy": ["spacy"],
}


# ---------------------------------------------------------------- PDF
def extract_text(pdf_file) -> str:
    """Extract text from a PDF (file path or Streamlit uploaded file)."""
    import pdfplumber

    parts = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


# ----------------------------------------------------- Preprocessing
_STOPWORDS = None
_LEMMATIZER = None


def _setup_nlp():
    """Prepare NLTK stopwords and lemmatizer (with a fallback if it fails)."""
    global _STOPWORDS, _LEMMATIZER
    if _STOPWORDS is not None:
        return
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        for pkg in ("stopwords", "wordnet"):
            nltk.download(pkg, quiet=True)
        _STOPWORDS = set(stopwords.words("english"))
        _LEMMATIZER = WordNetLemmatizer()
        _LEMMATIZER.lemmatize("test")  # checks that wordnet loaded correctly
    except Exception:
        _STOPWORDS = set(ENGLISH_STOP_WORDS)
        _LEMMATIZER = None


def preprocess(text: str) -> str:
    """Lowercase, tokenize, remove stopwords and lemmatize."""
    _setup_nlp()
    tokens = re.findall(r"[a-z0-9+#.]+", text.lower())
    tokens = [t.strip(".") for t in tokens]
    tokens = [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]
    if _LEMMATIZER:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# -------------------------------------------------------------- Embeddings
def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, size: int = 150) -> list:
    """Split text into chunks (the model only reads ~256 tokens at a time)."""
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def get_embedding(model, text: str) -> np.ndarray:
    """Embedding of the whole text = mean of the chunk embeddings."""
    vectors = model.encode(chunk_text(text))
    return np.mean(vectors, axis=0)


# ------------------------------------------------------------------ Skills
def extract_skills(text: str) -> set:
    """Return the set of known skills found in the text."""
    text = text.lower()
    found = set()
    for skill, variants in SKILLS.items():
        for v in variants:
            pattern = r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                found.add(skill)
                break
    return found


# ---------------------------------------------------------------- Extras
def extract_email(text: str) -> str:
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return m.group(0) if m else ""


def guess_name(text: str, fallback: str) -> str:
    """Try to use the first line of the resume as the candidate name."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line.split()) <= 4 and not re.search(r"[\d@]", line):
            return line.title()
        break
    return fallback.rsplit(".", 1)[0]


# ----------------------------------------------------------------- Ranking
def rank_candidates(model, resumes: dict, jd: str, weight_semantic: float = 0.6):
    """
    resumes: {file_name: text}
    Returns a DataFrame sorted from best to worst candidate.
    """
    names = list(resumes.keys())
    texts = [resumes[n] for n in names]

    # 1) Semantic similarity (embeddings)
    jd_emb = get_embedding(model, jd).reshape(1, -1)
    semantic_scores = []
    for t in texts:
        emb = get_embedding(model, t).reshape(1, -1)
        semantic_scores.append(float(cosine_similarity(emb, jd_emb)[0][0]))
    semantic_scores = np.clip(semantic_scores, 0, 1)

    # 2) TF-IDF baseline (keyword matching)
    corpus = [preprocess(jd)] + [preprocess(t) for t in texts]
    tfidf = TfidfVectorizer().fit_transform(corpus)
    tfidf_scores = cosine_similarity(tfidf[1:], tfidf[0]).flatten()

    # 3) Skills
    jd_skills = extract_skills(jd)
    rows = []
    for i, name in enumerate(names):
        found = extract_skills(texts[i])
        matched = sorted(found & jd_skills)
        missing = sorted(jd_skills - found)
        coverage = len(matched) / len(jd_skills) if jd_skills else None

        if coverage is None:
            final = semantic_scores[i]
        else:
            final = weight_semantic * semantic_scores[i] + (1 - weight_semantic) * coverage

        rows.append({
            "Candidate": guess_name(texts[i], name),
            "File": name,
            "Email": extract_email(texts[i]),
            "Final Score (%)": round(final * 100, 1),
            "Semantic (%)": round(semantic_scores[i] * 100, 1),
            "TF-IDF (%)": round(tfidf_scores[i] * 100, 1),
            "Skill Coverage (%)": round(coverage * 100, 1) if coverage is not None else None,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing),
        })

    df = pd.DataFrame(rows).sort_values("Final Score (%)", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)


# ------------------------------------------------------ Feedback (GenAI)
def rule_based_feedback(row) -> str:
    """Simple feedback without an API, used as a fallback."""
    score = row["Final Score (%)"]
    if score >= 70:
        recommendation = "Recommended for interview."
    elif score >= 50:
        recommendation = "Possible candidate; worth a manual review."
    else:
        recommendation = "Profile is not well aligned with the job."
    strengths = row["Matched Skills"] or "no skills from the job description found"
    missing = row["Missing Skills"] or "none"
    return (
        f"**Strengths:** {strengths}\n\n"
        f"**Missing skills:** {missing}\n\n"
        f"**Recommendation:** {recommendation}"
    )


def generate_ai_feedback(row, resume_text: str, jd: str, api_key: str = "") -> str:
    """Generate feedback with the Groq API. Without a key, use the fallback."""
    if not api_key:
        return rule_based_feedback(row)
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        prompt = (
            "You are a recruiting assistant. Compare the resume with the job "
            "description and answer briefly with: "
            "1) strengths, 2) skill gaps, 3) final recommendation "
            "(advance, maybe, or do not advance) with a short justification. "
            "Be objective and do not use personal information such as age, "
            "gender or origin.\n\n"
            f"JOB DESCRIPTION:\n{jd[:3000]}\n\nRESUME:\n{resume_text[:4000]}\n\n"
            f"Matched skills: {row['Matched Skills']}\n"
            f"Missing skills: {row['Missing Skills']}\n"
            f"Match score: {row['Final Score (%)']}%"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return rule_based_feedback(row) + f"\n\n_(AI unavailable: {e})_"