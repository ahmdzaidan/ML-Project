# file: utils.py
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file_bytes):
    import io
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                text += extract + " "
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def get_tfidf_score(text1, text2):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform([text1, text2])
        return cosine_similarity(matrix)[0][1] * 100
    except:
        return 0.0

def get_semantic_score(text1, text2):
    try:
        embedding1 = semantic_model.encode(text1, convert_to_tensor=True)
        embedding2 = semantic_model.encode(text2, convert_to_tensor=True)
        
        score = util.cos_sim(embedding1, embedding2)
        return score.item() * 100
    except:
        return 0.0

def calculate_match_score(resume_text, job_desc_text):
    if not resume_text or not job_desc_text:
        return 0.0
    
    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_desc_text)
    
    if not clean_resume or not clean_job:
        return 0.0

    tfidf_score = get_tfidf_score(clean_resume, clean_job)
    semantic_score = get_semantic_score(clean_resume, clean_job)

    # 30% TFIDF + 70% Semantic
    final_score = (tfidf_score * 0.3) + (semantic_score * 0.7)
    
    # Debugging
    print(f"--- DETAIL SKOR ---")
    print(f"Keyword Score (TF-IDF): {tfidf_score:.2f}%")
    print(f"Semantic Score (SBERT): {semantic_score:.2f}%")
    print(f"Final Hybrid Score    : {final_score:.2f}%")
    print(f"-------------------")

    return round(final_score, 2)