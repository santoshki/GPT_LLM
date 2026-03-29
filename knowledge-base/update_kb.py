import requests
from bs4 import BeautifulSoup
import json
import re
import math
from urllib.parse import urlparse
from collections import Counter
from difflib import SequenceMatcher


# ---------------------------------
# CONFIG
# ---------------------------------

KNOWLEDGE_FILE = "kb_healthcare.json"


# ---------------------------------
# 1️⃣ Fetch and Clean Web Content
# ---------------------------------

def fetch_web_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch URL: {response.status_code}")

    soup = BeautifulSoup(response.text, "lxml")

    # Remove junk sections
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)

    # Clean text
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"cookie|privacy|terms|subscribe|advertisement", "", text, flags=re.I)

    return text.strip()


# ---------------------------------
# 2️⃣ Accurate Summarization
# TF-IDF + Position + Definition Focus
# ---------------------------------

def summarize_text(text, num_sentences=6):
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if 40 < len(s) < 300]

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Prefer definition-style sentences
    definition_keywords = {" is ", " refers ", " defined ", " means ", " describes "}
    def_sentences = [
        s for s in sentences
        if any(k in s.lower() for k in definition_keywords)
    ]

    if len(def_sentences) >= num_sentences // 2:
        sentences = def_sentences + sentences

    # Word frequency
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(w for w in words if len(w) > 3)

    # Document frequency
    sentence_count = len(sentences)
    word_doc_freq = Counter()

    for sentence in sentences:
        unique_words = set(re.findall(r'\w+', sentence.lower()))
        for w in unique_words:
            word_doc_freq[w] += 1

    # IDF
    word_idf = {
        w: math.log(sentence_count / (1 + word_doc_freq[w]))
        for w in word_freq
    }

    # Sentence scoring
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\w+', sentence.lower())

        tfidf_score = sum(
            word_freq.get(w, 0) * word_idf.get(w, 0)
            for w in sentence_words
        )

        position_bonus = 1 / (1 + i)
        sentence_scores[sentence] = tfidf_score + position_bonus

    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    selected = sorted(ranked[:num_sentences], key=lambda s: sentences.index(s))

    return " ".join(selected)


# ---------------------------------
# 3️⃣ Smart Keyword Extraction
# ---------------------------------

STOPWORDS = {
    "this","that","with","from","have","been","were","their","there",
    "about","your","more","than","such","also","into","them","they",
    "what","when","where","which","while","will","would","could",
    "should","these","those","because","over","under","very"
}

def extract_keywords(text, top_n=8):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]


# ---------------------------------
# 4️⃣ Knowledge Base Functions
# ---------------------------------

def load_kb():
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_kb(data):
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_duplicate(new_answer, kb):
    for entry in kb:
        similarity = SequenceMatcher(None, new_answer, entry["answer"]).ratio()
        if similarity > 0.85:
            return True
    return False


# ---------------------------------
# 5️⃣ Split Summary into 2 KB Entries
# ---------------------------------

def split_summary(summary):
    sentences = re.split(r'(?<=[.!?]) +', summary)
    mid = len(sentences) // 2
    part1 = " ".join(sentences[:mid]).strip()
    part2 = " ".join(sentences[mid:]).strip()
    return part1, part2


def add_dual_entries(title, summary, keywords):
    kb = load_kb()

    part1, part2 = split_summary(summary)

    entries = [
        {
            "question": f"What is {title}?",
            "answer": part1,
            "keywords": keywords
        },
        {
            "question": f"What are important tips for {title}?",
            "answer": part2,
            "keywords": keywords
        }
    ]

    added = 0
    for new_entry in entries:
        if not is_duplicate(new_entry["answer"], kb):
            kb.append(new_entry)
            added += 1

    save_kb(kb)
    print(f"Added {added} new knowledge entries.")


# ---------------------------------
# 6️⃣ Title Extraction
# ---------------------------------

def extract_title_from_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string.strip()
        title = re.sub(r"\s*[-|–].*", "", title)
        return title
    except:
        return extract_title_from_url(url)


def extract_title_from_url(url):
    path = urlparse(url).path
    title = path.strip("/").split("/")[-1]
    title = title.replace("-", " ").replace("_", " ")
    return title.capitalize() if title else "Web Article"


# ---------------------------------
# 7️⃣ Main Pipeline
# ---------------------------------

def update_knowledge_from_url(url):
    print(f"Scraping: {url}")

    raw_text = fetch_web_content(url)

    if len(raw_text) < 500:
        raise Exception("Not enough content extracted. Page may block scraping.")

    print("Summarizing...")
    summary = summarize_text(raw_text)

    print("Extracting keywords...")
    keywords = extract_keywords(summary)

    title = extract_title_from_page(url).lower()

    print("Updating knowledge base...")
    add_dual_entries(title, summary, keywords)

    print("Done!")


# ---------------------------------
# Example Usage
# ---------------------------------

if __name__ == "__main__":
    url = "https://www.coreenergetics.org/methods-for-managing-stress-over-the-season-of-holidays/?gad_source=1&gad_campaignid=22300250112&gbraid=0AAAAApJEfmbvMfDX3lD2fffFQCCyiPm6F&gclid=Cj0KCQiA2bTNBhDjARIsAK89wlGFzMrjlOgJV-BMWahKFLtZuDxhjn-DfOgU-fFOuHSYEk7WXKQyCwUaAkKAEALw_wcB"
    update_knowledge_from_url(url)