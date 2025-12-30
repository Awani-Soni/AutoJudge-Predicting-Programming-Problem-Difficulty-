# streamlit_acm_app.py
import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack
import pandas as pd

# Load models and transformers

best_lr = joblib.load(
    r"D:\AWANI DOCUMENTS\ACM_Project\models\logreg_classifier.pkl")
gb_final = joblib.load(
    r"D:\AWANI DOCUMENTS\ACM_Project\models\gb_regressor.pkl")
tfidf = joblib.load(r"D:\AWANI DOCUMENTS\ACM_Project\models\tfidf.pkl")
le = joblib.load(r"D:\AWANI DOCUMENTS\ACM_Project\models\label_encoder.pkl")

# Functions 
def combine_text_columns(description, input_desc, output_desc):
    return description + " " + input_desc + " " + output_desc

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_numeric_features(text):
    text_lower = text.lower()

    algo_groups = {
        "dp": ["dp", "dynamic programming", "knapsack", "bitmask dp",
               "state", "transition", "memoization", "tabulation"],
        "graph": ["graph", "bfs", "dfs", "dijkstra", "bellman ford",
                  "floyd warshall", "topological", "shortest path",
                  "flow", "max flow", "min cut", "matching",
                  "strongly connected", "bridges", "articulation points",
                  "tree", "lca"],
        "ds": ["segment tree", "fenwick", "binary indexed tree",
               "heap", "priority queue", "stack", "queue",
               "deque", "union find", "disjoint set", "sparse table"],
        "math": ["modulo", "prime", "gcd", "lcm", "combinatorics",
                 "permutations", "probability", "matrix exponentiation",
                 "fft", "fast fourier transform", "number theory"],
        "geometry": ["geometry", "convex hull", "sweep line",
                     "cross product", "dot product", "orientation"],
        "string": ["string", "substring", "palindrome",
                   "kmp", "z algorithm", "suffix array",
                   "trie", "rolling hash"],
        "greedy": ["greedy", "two pointers", "sliding window",
                   "interval", "activity selection"]
    }

    group_counts = {}
    for group, keywords in algo_groups.items():
        count = sum(text_lower.count(k) for k in keywords)
        group_counts[f"{group}_count"] = np.log1p(count)

    math_symbols = "+-*/^=<>(){}[]|&!%"
    math_symbol_count = sum(text.count(sym) for sym in math_symbols)

    text_len = len(text)

    has_constraints = int(
        "≤" in text or "<=" in text_lower or "constraints" in text_lower)
    has_big_n = int(
        "10^5" in text or "10^6" in text or "10^7" in text or "10^" in text)
    has_time_limit = int("time limit" in text_lower or "seconds" in text_lower)

    return {
        "text_length": np.log1p(text_len),
        "math_symbol_count": np.log1p(math_symbol_count),
        "has_constraints": has_constraints,
        "has_big_n": has_big_n,
        "has_time_limit": has_time_limit,
        **group_counts
    }

# Streamlit UI

st.set_page_config(
    page_title="AutoJudge",
    page_icon="⚖️",
    layout="centered"
)

st.markdown(
    """
    <h1 style='
        text-align: center;
        margin-bottom: 0px;
        line-height: 1.2;
    '>
        Auto Judge
    </h1>

    <h2 style='
        text-align: center;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.4rem;
        font-weight: bold;
        line-height: 1.3;
    '>
        Predicting Programming Problem Difficulty
    </h2>

    <p style='text-align: center; color: grey; margin-top: 0;'>
        Paste the problem details below and click <b>Predict</b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Input section
st.markdown("### 📝 Problem Details")

description = st.text_area(
    "Problem Description",
    height=200,
    placeholder="Enter the full problem statement here..."
)

input_desc = st.text_area(
    "Input Description",
    height=120,
    placeholder="Give the input description..."
)

output_desc = st.text_area(
    "Output Description",
    height=120,
    placeholder="Give the expected output description..."
)

st.markdown("<br>", unsafe_allow_html=True)

# Centered Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button(
        "📋 Predict Difficulty", use_container_width=True)

# Prediction logic
if predict_clicked:
    if not description or not input_desc or not output_desc:
        st.warning("⚠️ Please fill in all fields before predicting.")
    else:
        # Combine + clean
        full_text = combine_text_columns(description, input_desc, output_desc)
        full_text_clean = clean_text(full_text)

        # TF-IDF features
        tfidf_features = tfidf.transform([full_text_clean])

        # Numeric features
        numeric_dict = extract_numeric_features(full_text_clean)
        numeric_df = pd.DataFrame([numeric_dict])

        # Final feature matrix
        X_final = hstack([tfidf_features, numeric_df.values])

        # Predictions
        class_encoded = best_lr.predict(X_final)[0]
        class_pred = le.inverse_transform([class_encoded])[0]
        score_pred = gb_final.predict(X_final)[0]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### ᯓ➤ Prediction Results")

        colA, colB = st.columns(2)

        with colA:
            st.success(f"**Difficulty Class**\n\n{class_pred.upper()}")

        with colB:
            st.info(f"**Difficulty Score**\n\n{score_pred:.2f}")

        st.markdown(
            "<p style='text-align:center; color:grey;'>"
            "Note: Score is an approximate difficulty estimation based on learned patterns."
            "</p>",
            unsafe_allow_html=True
        )