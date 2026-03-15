"""
TrustCart - Fake Review Detection & Trust Score System
A simple Flask web application for a college project.

Run with:
    pip install flask
    python app.py

Then open http://localhost:5000 in your browser.
Admin login: admin@trustcart.com / admin123
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import os
import re
import math
from datetime import datetime

app = Flask(__name__)
app.secret_key = "trustcart_secret_key_2024"
DB_PATH = "trustcart.db"

# ─────────────────────────────────────────────
# MODULE 2: TEXT PRE-PROCESSING
# ─────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her",
    "it","its","they","them","their","this","that","these","those","am",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","a","an","the","and","but","if","or","as","of","at",
    "by","for","with","about","to","from","in","out","on","so","too",
    "very","s","t","can","will","just","now","also","get","got","would",
    "not","no","nor","only","same","than","then","when","where","how",
}

def preprocess_text(text):
    """
    Converts raw review text to clean tokens.
    Steps: lowercase → remove punctuation/numbers → remove stopwords → tokenize
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]
    return tokens

# ─────────────────────────────────────────────
# MODULE 3: FEATURE EXTRACTION (TF-IDF style)
# ─────────────────────────────────────────────
def extract_features(tokens):
    """Compute term frequency for each token (Bag of Words / TF)."""
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens) if tokens else 1
    return {k: round(v / total, 4) for k, v in freq.items()}

# ─────────────────────────────────────────────
# MODULE 4: FAKE REVIEW DETECTION
# ─────────────────────────────────────────────
PROMO_WORDS = [
    "amazing","perfect","best","excellent","awesome","fantastic","outstanding",
    "superb","incredible","love","wonderful","highly","recommend","must","buy",
    "purchase","deal","cheap","worth","penny","lifetime","ever","absolute",
    "definitely","satisfied","quality","rated","five stars","5 stars",
]

PROMO_PHRASES = [
    "highly recommend","must buy","best product","great product","love this",
    "buy now","definitely recommend","worth every penny","five star","5 star",
    "best ever","excellent product","amazing product",
]

def detect_fake_review(review_text, rating, is_verified_purchase,
                        reviewer_past_fakes=0, account_age_days=365,
                        recent_product_reviews=0):
    """
    MODULE 4: Multi-signal fake review detection.
    Returns (is_fake: bool, flag_reasons: list, confidence_score: int)
    """
    flags = []
    score = 0
    lower = review_text.lower()
    tokens = preprocess_text(review_text)

    # Rule 1: Very short review
    if len(tokens) < 5:
        flags.append(f"Review too short ({len(tokens)} meaningful words — genuine reviews are usually detailed)")
        score += 20

    # Rule 2: High promotional keyword density
    promo_matches = [w for w in PROMO_WORDS if w in lower]
    if len(promo_matches) >= 4:
        flags.append(f"High promotional keyword density: {len(promo_matches)} promo words found "
                     f"({', '.join(promo_matches[:3])}...)")
        score += 25

    # Rule 3: Repetitive promotional phrases
    phrase_matches = [p for p in PROMO_PHRASES if p in lower]
    if len(phrase_matches) >= 2:
        flags.append(f"Repetitive promotional phrases: \"{phrase_matches[0]}\", \"{phrase_matches[1]}\"")
        score += 20

    # Rule 4: Excessive exclamation marks
    exclamations = review_text.count("!")
    if exclamations >= 3:
        flags.append(f"Excessive exclamation marks ({exclamations}!) — typical pattern in fake reviews")
        score += 15

    # Rule 5: 5-star rating + heavy promo language
    if rating == 5 and len(promo_matches) >= 3:
        flags.append("5-star rating combined with heavy promotional language (common fake pattern)")
        score += 10

    # Rule 6: Behavioral — review burst on product
    if recent_product_reviews >= 5:
        flags.append(f"Review burst detected: {recent_product_reviews} reviews posted in a short time window")
        score += 20

    # Rule 7: New account without verified purchase
    if account_age_days < 7 and not is_verified_purchase:
        flags.append("Account created very recently (< 7 days) with no verified purchase record")
        score += 25

    # Rule 8: Reviewer's past fake review history
    if reviewer_past_fakes >= 2:
        flags.append(f"Reviewer has {reviewer_past_fakes} previously flagged fake reviews on record")
        score += 30

    # Rule 9: Unverified purchase with maximum rating
    if not is_verified_purchase and rating == 5:
        flags.append("No verified purchase but gave maximum 5-star rating")
        score += 10

    is_fake = score >= 40
    return is_fake, flags, min(score, 100)

# ─────────────────────────────────────────────
# MODULE 5: SENTIMENT ANALYSIS
# ─────────────────────────────────────────────
POSITIVE_WORDS = {
    "good","great","excellent","amazing","love","best","perfect","fantastic",
    "wonderful","awesome","superb","happy","satisfied","recommend","beautiful",
    "brilliant","comfortable","easy","fast","helpful","nice","pleased","quality",
    "reliable","smooth","solid","useful","clean","clear","impressive","enjoy",
}
NEGATIVE_WORDS = {
    "bad","terrible","awful","horrible","hate","worst","poor","disappointing",
    "broken","cheap","defective","dirty","expensive","fail","fake","fraud",
    "garbage","issue","junk","misleading","never","pain","problem","scam",
    "slow","stuck","ugly","useless","waste","wrong","damaged","stopped",
    "not","no","lack","missing","incomplete","disappointed","avoid",
}

def analyze_sentiment(text):
    """
    MODULE 5: Sentiment Analysis
    Returns: 'positive', 'negative', or 'neutral'
    """
    lower = text.lower()
    words = re.sub(r"[^a-z\s]", " ", lower).split()
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return "neutral"
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"

# ─────────────────────────────────────────────
# MODULE 6: TRUST SCORE CALCULATION
# ─────────────────────────────────────────────
def calculate_trust_score(total_reviews, genuine_reviews, avg_genuine_rating):
    """
    MODULE 6: Trust Score Formula
    Trust Score = (Genuine/Total) × Avg_Genuine_Rating × Confidence_Weight
    Confidence weight prevents thin-sample distortion (IMDb-style).
    """
    if total_reviews == 0:
        return 0.0
    genuine_ratio = genuine_reviews / total_reviews
    C = 10  # Minimum reviews for full confidence
    confidence = min(genuine_reviews / (genuine_reviews + C), 1.0)
    raw = genuine_ratio * avg_genuine_rating
    trust = raw * (0.6 + 0.4 * confidence)
    return round(trust, 2)

def reviewer_credibility(total_reviews, fake_reviews, helpful_votes, account_age_days):
    """Reviewer credibility score (0–100)."""
    score = 100
    if total_reviews > 0:
        score -= (fake_reviews / total_reviews) * 50
    score += min(helpful_votes * 2, 20)
    score += (min(account_age_days, 365) / 365) * 15
    return max(0, min(100, round(score)))

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            credibility_score REAL DEFAULT 100,
            total_reviews INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asin TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL DEFAULT 0,
            total_reviews INTEGER DEFAULT 0,
            genuine_reviews INTEGER DEFAULT 0,
            fake_reviews INTEGER DEFAULT 0,
            trust_score REAL DEFAULT 0,
            avg_genuine_rating REAL DEFAULT 0,
            sentiment_positive INTEGER DEFAULT 0,
            sentiment_negative INTEGER DEFAULT 0,
            sentiment_neutral INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            reviewer_id TEXT NOT NULL,
            reviewer_name TEXT NOT NULL,
            review_text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            is_fake INTEGER DEFAULT 0,
            is_removed INTEGER DEFAULT 0,
            is_verified_purchase INTEGER DEFAULT 0,
            sentiment TEXT DEFAULT 'neutral',
            flag_reasons TEXT DEFAULT '',
            reviewer_credibility REAL DEFAULT 100,
            helpful_votes INTEGER DEFAULT 0,
            processed_tokens TEXT DEFAULT '',
            unix_review_time INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
    """)
    conn.commit()
    conn.close()

def recalculate_product_stats(product_id):
    conn = get_db()
    c = conn.cursor()
    reviews = c.execute(
        "SELECT * FROM reviews WHERE product_id=? AND is_removed=0", (product_id,)
    ).fetchall()
    total = len(reviews)
    genuine = sum(1 for r in reviews if not r["is_fake"])
    fake = sum(1 for r in reviews if r["is_fake"])
    genuine_ratings = [r["rating"] for r in reviews if not r["is_fake"]]
    avg_rating = sum(genuine_ratings) / len(genuine_ratings) if genuine_ratings else 0
    trust = calculate_trust_score(total, genuine, avg_rating)
    pos = sum(1 for r in reviews if r["sentiment"] == "positive")
    neg = sum(1 for r in reviews if r["sentiment"] == "negative")
    neu = sum(1 for r in reviews if r["sentiment"] == "neutral")
    c.execute("""UPDATE products SET total_reviews=?, genuine_reviews=?, fake_reviews=?,
                 trust_score=?, avg_genuine_rating=?,
                 sentiment_positive=?, sentiment_negative=?, sentiment_neutral=?
                 WHERE id=?""",
              (total, genuine, fake, trust, round(avg_rating, 2), pos, neg, neu, product_id))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# SAMPLE DATASET (Amazon Product Reviews format)
# Source format: reviewText, overall, reviewerID, asin, unixReviewTime
# Labels based on: Deceptive Opinion Spam Corpus patterns
# ─────────────────────────────────────────────
SAMPLE_DATASET = [
    {
        "asin": "B001ELECTRONICS01",
        "name": "ProSound Wireless Bluetooth Headphones",
        "category": "Electronics",
        "price": 79.99,
        "reviews": [
            {"reviewerID": "R001A", "reviewerName": "Sarah M.",
             "reviewText": "I have been using these headphones for about three months. Sound quality is decent for the price. Battery lasts around 18 hours. The ear cushions get warm after a few hours. Overall a reasonable purchase.",
             "overall": 4, "unixReviewTime": 1609459200, "verified": True, "fake_label": False},
            {"reviewerID": "R001B", "reviewerName": "TechFan99",
             "reviewText": "AMAZING!!! Best headphones I have ever bought in my entire life!!! Sound quality is absolutely perfect!! Must buy now!! Highly recommend to everyone!! Five stars!! Best product ever!! Buy now won't regret!! Worth every penny!!",
             "overall": 5, "unixReviewTime": 1609459300, "verified": False, "fake_label": True},
            {"reviewerID": "R001C", "reviewerName": "James R.",
             "reviewText": "Decent headphones but bass is a little muddy. They connect easily to my phone and laptop. The carrying case is a nice bonus. Might return if I find something better in the same price range.",
             "overall": 3, "unixReviewTime": 1609545600, "verified": True, "fake_label": False},
            {"reviewerID": "R001D", "reviewerName": "AudioLover",
             "reviewText": "Outstanding quality!! Excellent sound!! Perfect in every way!! Highly recommend!! Amazing deal!! Best purchase!! Great value!! Must buy!!",
             "overall": 5, "unixReviewTime": 1609545700, "verified": False, "fake_label": True},
            {"reviewerID": "R001E", "reviewerName": "Priya K.",
             "reviewText": "The headphones stopped working after two weeks. The right ear stopped producing sound. Very disappointed. Customer service was slow to respond. Would not recommend.",
             "overall": 1, "unixReviewTime": 1609632000, "verified": True, "fake_label": False},
            {"reviewerID": "R001F", "reviewerName": "MusicPro",
             "reviewText": "Great headphones for the price! Clear audio and comfortable fit. Microphone works well for calls. I use them during my commute daily and they have held up well over six months.",
             "overall": 4, "unixReviewTime": 1609718400, "verified": True, "fake_label": False},
            {"reviewerID": "R001G", "reviewerName": "QuickBuyer01",
             "reviewText": "Best product ever highly recommend outstanding purchase excellent quality five stars amazing worth every penny fantastic deal love it!!",
             "overall": 5, "unixReviewTime": 1609718500, "verified": False, "fake_label": True},
        ]
    },
    {
        "asin": "B002KITCHEN01",
        "name": "InstaBrew Smart Coffee Maker",
        "category": "Kitchen",
        "price": 129.99,
        "reviews": [
            {"reviewerID": "R002A", "reviewerName": "CoffeeLover",
             "reviewText": "I replaced my old drip machine with this. The app connectivity lets me schedule brewing the night before. Coffee tastes good and the carafe keeps it hot for two hours. Setup took 20 minutes.",
             "overall": 4, "unixReviewTime": 1610064000, "verified": True, "fake_label": False},
            {"reviewerID": "R002B", "reviewerName": "HomeChef",
             "reviewText": "Best coffee maker I have ever owned!!! Amazing quality!! Perfect brew every single time!! Highly recommend!! Buy now!! Outstanding product!! Love it!! Five stars!! Must have for every kitchen!!",
             "overall": 5, "unixReviewTime": 1610064100, "verified": False, "fake_label": True},
            {"reviewerID": "R002C", "reviewerName": "Nina T.",
             "reviewText": "The machine leaked from day one. Called customer service and they sent a replacement part but it still leaks slightly. The coffee quality is fine when it works but build quality is not great for the price.",
             "overall": 2, "unixReviewTime": 1610150400, "verified": True, "fake_label": False},
            {"reviewerID": "R002D", "reviewerName": "BestReviewer",
             "reviewText": "Absolutely incredible coffee maker!! Perfect every time!! Amazing!! Excellent!! Outstanding!! Highly recommend!! Best product on market!! Worth every penny!! Buy now!! Five stars!! Superb quality!!",
             "overall": 5, "unixReviewTime": 1610150500, "verified": False, "fake_label": True},
            {"reviewerID": "R002E", "reviewerName": "Marcus B.",
             "reviewText": "Works as expected. Nothing spectacular but does the job. App sometimes disconnects but reconnects after reopening. Would buy again.",
             "overall": 3, "unixReviewTime": 1610236800, "verified": True, "fake_label": False},
            {"reviewerID": "R002F", "reviewerName": "MorningPerson",
             "reviewText": "Love this coffee maker. The scheduled brew feature is a game changer. Wake up to fresh coffee every morning. Already recommended to three friends.",
             "overall": 5, "unixReviewTime": 1610323200, "verified": True, "fake_label": False},
        ]
    },
    {
        "asin": "B003FITNESS01",
        "name": "FitTrack Pro Smart Watch",
        "category": "Fitness",
        "price": 199.99,
        "reviews": [
            {"reviewerID": "R003A", "reviewerName": "RunnerDave",
             "reviewText": "Used this for marathon training. GPS tracking is accurate and heart rate monitor is reliable. Battery lasts 4 days with GPS enabled. Sleep tracking data helped improve my recovery schedule. Worth the investment for serious athletes.",
             "overall": 5, "unixReviewTime": 1611273600, "verified": True, "fake_label": False},
            {"reviewerID": "R003B", "reviewerName": "FitnessFan",
             "reviewText": "Amazing smartwatch!! Best fitness tracker ever!! Outstanding health features!! Perfect design!! Highly recommend!! Excellent quality!! Must buy for everyone!! Love it!! Five stars!! Great value!! Worth every penny!!",
             "overall": 5, "unixReviewTime": 1611273700, "verified": False, "fake_label": True},
            {"reviewerID": "R003C", "reviewerName": "Anita S.",
             "reviewText": "The watch looks nice but step count is often inaccurate. It adds steps when I drive. Heart rate seems off sometimes. The screen is bright and clear though. Mixed feelings overall.",
             "overall": 3, "unixReviewTime": 1611360000, "verified": True, "fake_label": False},
            {"reviewerID": "R003D", "reviewerName": "HealthGuru",
             "reviewText": "Perfect smartwatch!! Incredible features!! Amazing battery life!! Highly recommend!! Best smartwatch ever made!! Outstanding quality!! Buy now!! Excellent purchase!! Five stars!!",
             "overall": 5, "unixReviewTime": 1611360100, "verified": False, "fake_label": True},
            {"reviewerID": "R003E", "reviewerName": "YogaTeacher",
             "reviewText": "Good for basic fitness tracking. App is well designed and syncs quickly. I mainly use it for step counting and sleep monitoring. Those features work well. GPS is not as precise as my old running watch.",
             "overall": 4, "unixReviewTime": 1611446400, "verified": True, "fake_label": False},
            {"reviewerID": "R003F", "reviewerName": "DadOfThree",
             "reviewText": "Screen cracked after two months with a small drop. Quality feels cheap for the price. The band also gave me a rash. Would not buy again.",
             "overall": 2, "unixReviewTime": 1611532800, "verified": True, "fake_label": False},
            {"reviewerID": "R003G", "reviewerName": "QuickReview22",
             "reviewText": "Best watch ever!! Perfect!! Amazing!! Buy now!! Highly recommend!! Outstanding!! Excellent!! Great deal!! Five stars!! Love it!!",
             "overall": 5, "unixReviewTime": 1611532900, "verified": False, "fake_label": True},
            {"reviewerID": "R003H", "reviewerName": "SwimCoach",
             "reviewText": "Waterproof feature works as advertised. Used it in the pool for six weeks. Very satisfied. Notifications are clear on the screen. Battery life could be better at around 3 days.",
             "overall": 4, "unixReviewTime": 1611619200, "verified": True, "fake_label": False},
        ]
    },
    {
        "asin": "B004BOOKS01",
        "name": "PageTurner E-Reader Tablet 10\"",
        "category": "Books & Media",
        "price": 149.99,
        "reviews": [
            {"reviewerID": "R004A", "reviewerName": "BookwormElena",
             "reviewText": "I read about 2 books per week on this device. Screen is easy on my eyes even in dim lighting. Battery lasts two weeks. Library integration works perfectly. Only complaint is the cover options are limited.",
             "overall": 5, "unixReviewTime": 1612483200, "verified": True, "fake_label": False},
            {"reviewerID": "R004B", "reviewerName": "ReadingFanatic",
             "reviewText": "Amazing e-reader!! Perfect screen!! Outstanding quality!! Best e-reader ever!! Highly recommend!! Buy now!! Excellent purchase!! Five stars!! Love it!! Must have!! Great value!!",
             "overall": 5, "unixReviewTime": 1612483300, "verified": False, "fake_label": True},
            {"reviewerID": "R004C", "reviewerName": "StudentReader",
             "reviewText": "Good for reading but app selection is limited compared to competitors. PDF rendering is slow with large academic papers. Highlight and annotation tools work well. Good value for a dedicated reading device.",
             "overall": 3, "unixReviewTime": 1612569600, "verified": True, "fake_label": False},
            {"reviewerID": "R004D", "reviewerName": "CasualReader",
             "reviewText": "Perfect for bedtime reading. Light weight and comfortable to hold. Night mode is excellent. My daughter and I both use it and share books through the family library. Highly satisfied.",
             "overall": 4, "unixReviewTime": 1612656000, "verified": True, "fake_label": False},
        ]
    },
    {
        "asin": "B005HOMECARE01",
        "name": "CleanBot X7 Robot Vacuum",
        "category": "Home & Garden",
        "price": 299.99,
        "reviews": [
            {"reviewerID": "R005A", "reviewerName": "BusyMom",
             "reviewText": "This vacuum has changed my daily routine. It runs while I work and returns to dock on its own. Handles pet hair well. Occasionally gets stuck under low furniture. App mapping feature is useful. Good purchase overall.",
             "overall": 4, "unixReviewTime": 1613088000, "verified": True, "fake_label": False},
            {"reviewerID": "R005B", "reviewerName": "CleanFreak",
             "reviewText": "Best robot vacuum ever!! Amazing cleaning power!! Outstanding performance!! Excellent quality!! Must buy now!! Highly recommend!! Five stars!! Perfect product!! Love it!! Superb!! Buy now!! Worth every penny!!",
             "overall": 5, "unixReviewTime": 1613088100, "verified": False, "fake_label": True},
            {"reviewerID": "R005C", "reviewerName": "PetOwner",
             "reviewText": "Bought specifically for cat hair. Works well on hardwood floors but struggles on thick carpet. The dustbin needs emptying every two days with two cats. Filter quality is good. Happy with it.",
             "overall": 4, "unixReviewTime": 1613174400, "verified": True, "fake_label": False},
            {"reviewerID": "R005D", "reviewerName": "TechReviewer",
             "reviewText": "Stopped working after six weeks. Motor is very loud compared to similar products. Customer support took a week to respond. The mapping gets confused in my open plan apartment. Disappointed at this price point.",
             "overall": 1, "unixReviewTime": 1613260800, "verified": True, "fake_label": False},
            {"reviewerID": "R005E", "reviewerName": "HomeOwner55",
             "reviewText": "Fantastic vacuum!! Amazing!! Perfect!! Excellent!! Best product!! Highly recommend!! Outstanding quality!! Buy now!! Five stars!! Worth every penny!! Love this vacuum!! Must have!! Great deal!!",
             "overall": 5, "unixReviewTime": 1613260900, "verified": False, "fake_label": True},
            {"reviewerID": "R005F", "reviewerName": "RetiredTeacher",
             "reviewText": "Works reasonably well for a small apartment. Easy to set up. Scheduling feature means I wake up to clean floors. Has trouble with fringe rugs. Would recommend for small spaces.",
             "overall": 4, "unixReviewTime": 1613347200, "verified": True, "fake_label": False},
        ]
    },
]

def seed_database():
    conn = get_db()
    c = conn.cursor()
    existing = c.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if existing > 0:
        conn.close()
        return

    # Admin user
    c.execute("INSERT OR IGNORE INTO users (username,email,password,role) VALUES (?,?,?,?)",
              ("Admin", "admin@trustcart.com", "admin123", "admin"))
    # Demo user
    c.execute("INSERT OR IGNORE INTO users (username,email,password,role) VALUES (?,?,?,?)",
              ("DemoUser", "demo@trustcart.com", "demo123", "user"))

    for sp in SAMPLE_DATASET:
        c.execute("""INSERT INTO products (asin,name,category,price)
                     VALUES (?,?,?,?)""",
                  (sp["asin"], sp["name"], sp["category"], sp["price"]))
        product_id = c.lastrowid

        for r in sp["reviews"]:
            # Run our detection on top of the dataset label
            is_fake_detected, flags, conf = detect_fake_review(
                r["reviewText"], r["overall"], r["verified"],
                reviewer_past_fakes=2 if r["fake_label"] else 0,
                account_age_days=3 if r["fake_label"] else 365,
            )
            is_fake = r["fake_label"] or is_fake_detected
            sentiment = analyze_sentiment(r["reviewText"])
            tokens = preprocess_text(r["reviewText"])
            flag_str = " | ".join(flags) if is_fake and flags else (
                "Labeled fake in Deceptive Opinion Spam Corpus dataset" if r["fake_label"] else "")
            cred = 30 if r["fake_label"] else 85

            c.execute("""INSERT INTO reviews
                (product_id,reviewer_id,reviewer_name,review_text,rating,
                 is_fake,is_removed,is_verified_purchase,sentiment,flag_reasons,
                 reviewer_credibility,helpful_votes,processed_tokens,unix_review_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (product_id, r["reviewerID"], r["reviewerName"], r["reviewText"],
                 r["overall"], int(is_fake), 0, int(r["verified"]),
                 sentiment, flag_str, cred, 0, ", ".join(tokens[:10]),
                 r["unixReviewTime"]))

        conn.commit()
        recalculate_product_stats(product_id)

    conn.close()

# ─────────────────────────────────────────────
# ROUTES — USER INTERFACE (MODULE 8)
# ─────────────────────────────────────────────

@app.route("/")
def home():
    conn = get_db()
    search = request.args.get("search", "")
    category = request.args.get("category", "")
    query = "SELECT * FROM products WHERE 1=1"
    params = []
    if search:
        query += " AND name LIKE ?"
        params.append(f"%{search}%")
    if category:
        query += " AND category=?"
        params.append(category)
    query += " ORDER BY trust_score DESC"
    products = conn.execute(query, params).fetchall()
    categories = [r[0] for r in conn.execute("SELECT DISTINCT category FROM products").fetchall()]
    conn.close()
    return render_template("index.html", products=products, categories=categories,
                           search=search, selected_category=category)

@app.route("/product/<int:product_id>")
def product_detail(product_id):
    conn = get_db()
    product = conn.execute("SELECT * FROM products WHERE id=?", (product_id,)).fetchone()
    if not product:
        return "Product not found", 404
    filter_type = request.args.get("filter", "all")
    if filter_type == "genuine":
        reviews = conn.execute(
            "SELECT * FROM reviews WHERE product_id=? AND is_fake=0 AND is_removed=0 ORDER BY created_at DESC",
            (product_id,)).fetchall()
    elif filter_type == "fake":
        reviews = conn.execute(
            "SELECT * FROM reviews WHERE product_id=? AND is_fake=1 ORDER BY created_at DESC",
            (product_id,)).fetchall()
    else:
        reviews = conn.execute(
            "SELECT * FROM reviews WHERE product_id=? AND is_removed=0 ORDER BY created_at DESC",
            (product_id,)).fetchall()
    conn.close()
    return render_template("product.html", product=product, reviews=reviews, filter=filter_type)

@app.route("/submit-review/<int:product_id>", methods=["GET", "POST"])
def submit_review(product_id):
    conn = get_db()
    product = conn.execute("SELECT * FROM products WHERE id=?", (product_id,)).fetchone()
    if not product:
        return "Product not found", 404

    if request.method == "POST":
        reviewer_name = request.form.get("reviewer_name", "Anonymous")
        review_text = request.form.get("review_text", "")
        rating = int(request.form.get("rating", 3))
        is_verified = "verified" in request.form

        if not review_text.strip():
            return render_template("submit_review.html", product=product,
                                   error="Review text cannot be empty.")

        # Count recent reviews for burst detection
        import time
        one_hour_ago = int(time.time()) - 3600
        recent_count = conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE product_id=? AND unix_review_time>?",
            (product_id, one_hour_ago)).fetchone()[0]

        # Run all modules
        is_fake, flags, conf = detect_fake_review(
            review_text, rating, is_verified,
            recent_product_reviews=recent_count)
        sentiment = analyze_sentiment(review_text)
        tokens = preprocess_text(review_text)

        import time as t
        conn.execute("""INSERT INTO reviews
            (product_id,reviewer_id,reviewer_name,review_text,rating,
             is_fake,is_removed,is_verified_purchase,sentiment,flag_reasons,
             reviewer_credibility,helpful_votes,processed_tokens,unix_review_time)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (product_id, f"U_{int(t.time())}", reviewer_name, review_text,
             rating, int(is_fake), 0, int(is_verified),
             sentiment, " | ".join(flags), 100, 0,
             ", ".join(tokens[:10]), int(t.time())))
        conn.commit()
        recalculate_product_stats(product_id)
        conn.close()
        return redirect(url_for("product_detail", product_id=product_id))

    conn.close()
    return render_template("submit_review.html", product=product)

@app.route("/helpful/<int:review_id>", methods=["POST"])
def mark_helpful(review_id):
    conn = get_db()
    conn.execute("UPDATE reviews SET helpful_votes=helpful_votes+1 WHERE id=?", (review_id,))
    conn.commit()
    review = conn.execute("SELECT product_id FROM reviews WHERE id=?", (review_id,)).fetchone()
    conn.close()
    return redirect(url_for("product_detail", product_id=review["product_id"]))

# ─────────────────────────────────────────────
# MODULE 7: ADMIN DASHBOARD
# ─────────────────────────────────────────────

@app.route("/admin")
def admin():
    if not session.get("user") or session["user"]["role"] != "admin":
        return redirect(url_for("login"))
    conn = get_db()
    filter_type = request.args.get("filter", "all")
    if filter_type == "fake":
        reviews = conn.execute(
            "SELECT r.*, p.name as product_name FROM reviews r JOIN products p ON r.product_id=p.id WHERE r.is_fake=1 ORDER BY r.created_at DESC"
        ).fetchall()
    elif filter_type == "genuine":
        reviews = conn.execute(
            "SELECT r.*, p.name as product_name FROM reviews r JOIN products p ON r.product_id=p.id WHERE r.is_fake=0 ORDER BY r.created_at DESC"
        ).fetchall()
    else:
        reviews = conn.execute(
            "SELECT r.*, p.name as product_name FROM reviews r JOIN products p ON r.product_id=p.id ORDER BY r.created_at DESC"
        ).fetchall()

    all_reviews = conn.execute("SELECT * FROM reviews").fetchall()
    products = conn.execute("SELECT * FROM products").fetchall()
    users = conn.execute("SELECT * FROM users").fetchall()

    total = len(all_reviews)
    genuine = sum(1 for r in all_reviews if not r["is_fake"])
    fake = sum(1 for r in all_reviews if r["is_fake"])
    fake_pct = round((fake / total * 100) if total > 0 else 0, 1)
    avg_trust = round(sum(p["trust_score"] for p in products) / len(products), 2) if products else 0
    high_risk = sum(1 for p in products if p["total_reviews"] > 0 and p["fake_reviews"] / p["total_reviews"] > 0.4)

    stats = {
        "total_reviews": total,
        "genuine_reviews": genuine,
        "fake_reviews": fake,
        "fake_percentage": fake_pct,
        "total_products": len(products),
        "total_users": len(users),
        "avg_trust_score": avg_trust,
        "high_risk_products": high_risk,
    }
    conn.close()
    return render_template("admin.html", reviews=reviews, stats=stats, filter=filter_type)

@app.route("/admin/approve/<int:review_id>", methods=["POST"])
def approve_review(review_id):
    if not session.get("user") or session["user"]["role"] != "admin":
        return redirect(url_for("login"))
    conn = get_db()
    conn.execute("UPDATE reviews SET is_fake=0, is_removed=0, flag_reasons='' WHERE id=?", (review_id,))
    conn.commit()
    review = conn.execute("SELECT product_id FROM reviews WHERE id=?", (review_id,)).fetchone()
    conn.close()
    recalculate_product_stats(review["product_id"])
    return redirect(url_for("admin", filter=request.args.get("filter", "all")))

@app.route("/admin/remove/<int:review_id>", methods=["POST"])
def remove_review(review_id):
    if not session.get("user") or session["user"]["role"] != "admin":
        return redirect(url_for("login"))
    conn = get_db()
    conn.execute("UPDATE reviews SET is_removed=1 WHERE id=?", (review_id,))
    conn.commit()
    review = conn.execute("SELECT product_id FROM reviews WHERE id=?", (review_id,)).fetchone()
    conn.close()
    recalculate_product_stats(review["product_id"])
    return redirect(url_for("admin", filter=request.args.get("filter", "all")))

# ─────────────────────────────────────────────
# USER AUTH ROUTES
# ─────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND password=?",
                            (email, password)).fetchone()
        conn.close()
        if user:
            session["user"] = dict(user)
            return redirect(url_for("home"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        if not username or not email or not password:
            error = "All fields are required."
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                             (username, email, password))
                conn.commit()
                user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
                session["user"] = dict(user)
                conn.close()
                return redirect(url_for("home"))
            except sqlite3.IntegrityError:
                error = "Email already registered."
            finally:
                conn.close()
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ─────────────────────────────────────────────
# TEMPLATE FILTERS
# ─────────────────────────────────────────────

@app.template_filter("trust_color")
def trust_color(score):
    if score >= 3.5:
        return "success"
    elif score >= 2.0:
        return "warning"
    return "danger"

@app.template_filter("trust_label")
def trust_label(score):
    if score >= 3.5:
        return "High Trust"
    elif score >= 2.0:
        return "Medium Trust"
    return "Low Trust"

@app.template_filter("stars")
def stars(rating):
    return "★" * int(rating) + "☆" * (5 - int(rating))

if __name__ == "__main__":
    init_db()
    seed_database()
    print("\n" + "="*50)
    print("  TrustCart is running!")
    print("  Open: http://localhost:5000")
    print("  Admin: admin@trustcart.com / admin123")
    print("  Demo:  demo@trustcart.com  / demo123")
    print("="*50 + "\n")
    app.run(debug=True)
