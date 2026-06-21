# SkillSync — Tech Role Recommendation Engine

SkillSync is a content-based AI recommendation engine that matches user skills
to relevant tech job roles. Built using **TF-IDF feature extraction** and
**Cosine Similarity**, it analyzes an IT job roles dataset, ranks the best-fit
careers, and includes a cold-start fallback. Comes with full EDA, a Jupyter
notebook pipeline, and an interactive Streamlit GUI.

🔗 **Live App:** [skillsync-tech-role-recommendation-engine.streamlit.app](https://skillsync-tech-role-recommendation-engine.streamlit.app/)

## Project Structure

```
SkillSync-Tech-Role-Recommendation-Engine/
│
├── .streamlit/
│   └── config.toml              # Custom theme (Stormy / Cloud / Sunset / Evening palette)
│
├── dataset/
│   └── IT_Job_Roles_Skills.csv  # Source dataset
│
├── images/
│   ├── certification_pie.png
│   ├── description_length.png
│   ├── skills_per_role.png
│   └── top_skills.png
│
├── models/
│   ├── job_roles_processed.pkl  # Cleaned dataframe
│   ├── tfidf_matrix.joblib      # Pre-computed TF-IDF matrix
│   └── vectorizer.joblib        # Trained TF-IDF vectorizer
│
├── notebook/
│   └── Tech_Stack_Recommender.ipynb   # EDA + Feature Engineering + Recommendation Logic
│
├── App.py                       # Streamlit GUI
├── README.md
└── requirements.txt
```

## Features

- Full EDA on an IT job roles & skills dataset (skill frequency, role complexity, description analysis)
- TF-IDF based feature engineering that weighs rare/specific skills higher than generic ones
- Cosine Similarity-based ranking engine with Top-N recommendations
- Cold-start fallback for unmatched skill inputs
- Interactive Streamlit GUI with a custom color theme

## Tech Stack

Python · pandas · scikit-learn · Streamlit · Matplotlib/Seaborn

## How It Works

1. **Ingestion** — user enters their skills
2. **Scoring** — skills are converted into TF-IDF vectors and compared against every job role using Cosine Similarity
3. **Sorting** — roles are ranked by similarity score, descending
4. **Filtering** — only the Top-N best matches are returned
5. **Cold-Start Fallback** — if no skill overlap is found, popular roles are suggested instead

## Run Locally

1. Clone the repository and install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) Re-run the notebook to regenerate the model artifacts:
   ```
   jupyter notebook notebook/Tech_Stack_Recommender.ipynb
   ```

3. Launch the app:
   ```
   streamlit run App.py
   ```

4. Open the local URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## Try It Online

No installation needed — try the live deployed app here:
-> **https://skillsync-tech-role-recommendation-engine.streamlit.app/**

---

Built for **DecodeLabs Project 3: AI Recommendation Logic** (Industrial Training Kit, Batch 2026).