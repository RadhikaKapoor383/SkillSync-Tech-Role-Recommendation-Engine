"""
SkillSync — Tech Role Recommendation Engine (Streamlit GUI)
--------------------------------------------------------------
Run with:
    streamlit run App.py

Folder structure expected:
    dataset/  -> IT_Job_Roles_Skills.csv
    models/   -> vectorizer.joblib, tfidf_matrix.joblib, job_roles_processed.pkl
    notebook/ -> Tech_Stack_Recommender.ipynb
    App.py
"""

import re
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="SkillSync — Tech Role Recommender",
    page_icon=":material/explore:",
    layout="centered",
)

# -----------------------------------------------------------------------
# Custom styling — aligned with .streamlit/config.toml palette
#   Stormy  #494E6B   Cloud  #98878F
#   Sunset  #985E6D   Evening #192231
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
        h1, h2, h3 {
            color: #494E6B !important;   /* Stormy */
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #192231 !important;   /* Evening */
        }
        [data-testid="stMetricValue"] {
            color: #985E6D !important;   /* Sunset */
        }
        div[data-testid="stContainer"] {
            border-color: #494E6B !important;
        }
        hr {
            border-top: 1px solid #98878F !important;  /* Cloud */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# Load artifacts (cached so they only load once)
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("models/vectorizer.joblib")
    tfidf_matrix = joblib.load("models/tfidf_matrix.joblib")
    df = pd.read_pickle("models/job_roles_processed.pkl")
    return vectorizer, tfidf_matrix, df


vectorizer, tfidf_matrix, df = load_artifacts()

# Build the full sorted list of unique skills for the multiselect widget
all_skills_set = sorted({skill for skills in df["Skills_List"] for skill in skills})


def clean_skills_text(skills_list):
    cleaned = [
        re.sub(r"[^a-zA-Z0-9\s]", "", s).strip().lower().replace(" ", "_")
        for s in skills_list
    ]
    return " ".join(cleaned)


def recommend_jobs(user_skills, top_n=3):
    user_text = clean_skills_text(user_skills)
    user_vector = vectorizer.transform([user_text])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    df_temp = df.copy()
    df_temp["Similarity_Score"] = similarity_scores
    ranked = df_temp.sort_values("Similarity_Score", ascending=False)
    top_matches = ranked[ranked["Similarity_Score"] > 0].head(top_n)
    return top_matches.reset_index(drop=True)


def recommend_with_fallback(user_skills, top_n=3):
    results = recommend_jobs(user_skills, top_n=top_n)
    if results.empty:
        fallback = df.sample(top_n, random_state=42).reset_index(drop=True)
        fallback["Similarity_Score"] = 0.0
        return fallback, True
    return results, False


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title(":material/explore: SkillSync")
st.caption("AI-Powered Tech Role Recommendation Engine")
st.markdown(
    "Tell us your skills, and we'll recommend the **tech job roles** that match "
    "best — powered by **TF-IDF** + **Cosine Similarity**."
)

st.divider()

# -----------------------------------------------------------------------
# Input section
# -----------------------------------------------------------------------
st.subheader(":material/checklist: 1. Enter Your Skills")

input_mode = st.radio(
    "How would you like to enter your skills?",
    options=["Choose from list", "Type manually"],
    horizontal=True,
)

if input_mode == "Choose from list":
    selected_skills = st.multiselect(
        "Select your skills (choose at least 1):",
        options=all_skills_set,
        default=["Python", "Cloud Computing"] if "Python" in all_skills_set else None,
    )
else:
    raw_text = st.text_input(
        "Type your skills, comma-separated:",
        placeholder="e.g. Python, Docker, Kubernetes, AWS",
    )
    selected_skills = [s.strip() for s in raw_text.split(",") if s.strip()]

top_n = st.slider("How many recommendations do you want?", min_value=1, max_value=10, value=3)

st.subheader(":material/search: 2. Get Recommendations")
go = st.button(
    "Recommend Job Roles",
    type="primary",
    icon=":material/manage_search:",
    use_container_width=True,
)

st.divider()

# -----------------------------------------------------------------------
# Output section
# -----------------------------------------------------------------------
if go:
    if not selected_skills:
        st.warning(
            "Please enter at least one skill before requesting recommendations.",
            icon=":material/warning:",
        )
    else:
        results, used_fallback = recommend_with_fallback(selected_skills, top_n=top_n)

        st.subheader(":material/person: Your Skills")
        st.write(", ".join(selected_skills))

        if used_fallback:
            st.info(
                "No close match was found for those skills, so here are some "
                "popular roles to explore instead (Cold-Start fallback).",
                icon=":material/info:",
            )

        st.subheader(f":material/workspace_premium: Top {len(results)} Recommended Job Roles")

        for i, row in results.iterrows():
            score_pct = row["Similarity_Score"] * 100
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### {i + 1}. {row['Job Title']}")
                with col2:
                    st.metric("Match", f"{score_pct:.0f}%")

                st.progress(min(row["Similarity_Score"], 1.0))

                with st.expander("View required skills & description"):
                    st.markdown(f"**Skills:** {row['Skills']}")
                    if pd.notna(row.get("Job Description")):
                        st.markdown(f"**Description:** {row['Job Description']}")
                    if pd.notna(row.get("Certifications")):
                        st.markdown(f"**Suggested Certifications:** {row['Certifications']}")
else:
    st.caption(
        "Enter your skills above and click **Recommend Job Roles** to see results."
    )

# -----------------------------------------------------------------------
# Sidebar info
# -----------------------------------------------------------------------
with st.sidebar:
    st.header(":material/info: About SkillSync")
    st.markdown(
        "A content-based recommendation engine that matches your skills to "
        "the most relevant **tech job roles** — no historical user data required."
    )

    st.divider()

    st.subheader(":material/account_tree: How It Works")
    st.markdown(
        """
1. **Ingestion** — your selected skills are captured
2. **Scoring** — skills are converted into TF-IDF vectors and compared against every job role
3. **Sorting** — roles are ranked by Cosine Similarity score
4. **Filtering** — only the Top-N best matches are shown
5. **Cold-Start Fallback** — if no overlap is found, popular roles are suggested instead
        """
    )

    st.divider()

    st.subheader(":material/dataset: Dataset Snapshot")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Job Roles", len(df))
    with col_b:
        st.metric("Unique Skills", len(all_skills_set))

    avg_skills = df["Num_Skills"].mean() if "Num_Skills" in df.columns else None
    if avg_skills:
        st.metric("Avg. Skills per Role", f"{avg_skills:.1f}")

    st.divider()

    st.subheader(":material/build: Tech Stack")
    st.markdown(
        """
- **Python** — core logic
- **pandas** — data processing
- **scikit-learn** — TF-IDF & Cosine Similarity
- **Streamlit** — interactive GUI
        """
    )

    st.divider()

    st.caption("Built for **DecodeLabs Project 3: AI Recommendation Logic**")
    st.caption("SkillSync — Tech Role Recommendation Engine")