
import streamlit as st
import pandas as pd
import re
import nltk
import contractions

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(page_title="Storyline Recommendation", layout="wide")

# -----------------------
# Initialize session state
# -----------------------
if "best_match" not in st.session_state:
    st.session_state.best_match = None

if "top_df" not in st.session_state:
    st.session_state.top_df = None

# -----------------------
# NLTK setup
# -----------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------
# Text preprocessing
# -----------------------
def clean_text(text):
    if pd.isna(text):
        return ""

    text = contractions.fix(str(text))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("streamlit_storyline.csv")
df.columns = df.columns.str.strip().str.lower()

required_cols = ["title_clean", "final_clean_storyline"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Dataset must have '{col}' column!")
        st.stop()

df = df.dropna(subset=required_cols).reset_index(drop=True)
df["clean_storyline"] = df["final_clean_storyline"].apply(clean_text)

# -----------------------
# TF-IDF Vectorization
# -----------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["clean_storyline"])


st.title(" Storyline Recommendation System")

user_input = st.text_area("Enter your text/query here:")
top_n = st.slider("How many recommendations?", 1, 20, 5)


if st.button("Get Recommendations"):

    if user_input.strip() == "":
        st.warning("Please enter some text to get recommendations.")
    else:
        query = clean_text(user_input)
        query_vec = vectorizer.transform([query])

        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

        top_idx = similarity.argsort()[-top_n:][::-1]
        top_df = df.iloc[top_idx].copy()
        top_df["similarity_score"] = similarity[top_idx]

        # Save results to session state
        st.session_state.top_df = top_df
        st.session_state.best_match = top_df.iloc[0]

# -----------------------
# Display results (ONLY if available)
# -----------------------
if st.session_state.top_df is not None:

    st.subheader("Top Recommendations:")

    for i, (_, row) in enumerate(
        st.session_state.top_df.iterrows(), start=1
    ):
        st.markdown(f"### {i}. {row['title_clean']}")
        st.write(row["final_clean_storyline"])
        st.caption(f"Similarity Score: {row['similarity_score']:.4f}")
        st.write("---")


if st.session_state.best_match is not None:
    st.success(
        f" Best Match: **{st.session_state.best_match['title_clean']}**"
    )

    st.subheader("Similarity Score Comparison")
    chart_df = (
        st.session_state.top_df[
            ["title_clean", "similarity_score"]
        ]
        .set_index("title_clean")
    )
    st.bar_chart(chart_df)
