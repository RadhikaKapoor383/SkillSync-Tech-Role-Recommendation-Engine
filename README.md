# SkillSync — Tech Role Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-TF--IDF-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Recommendation%20Engine-green)

SkillSync is an AI-powered career recommendation engine that matches user skills with relevant technology job roles. Using **TF-IDF feature extraction** and **Cosine Similarity**, the system analyzes technical skill sets and recommends the most suitable career paths.

The project includes comprehensive exploratory data analysis, feature engineering, recommendation logic, and an interactive Streamlit web application.

**Live Demo:** https://skillsync-tech-role-recommendation-engine.streamlit.app/  </br>
**GitHub Repository:** https://github.com/RadhikaKapoor383/SkillSync-Tech-Role-Recommendation-Engine

---

## Project Overview

Choosing the right technology career path can be challenging for students and aspiring professionals. SkillSync addresses this problem by analyzing users' technical skills and recommending suitable job roles based on similarity to industry skill requirements.

The recommendation engine leverages Natural Language Processing (NLP) techniques to compare user-entered skills against real-world technology roles.

---

## Features

* AI-powered job role recommendation engine.
* TF-IDF feature extraction for skill representation.
* Cosine Similarity-based role matching.
* Top-N ranked career recommendations.
* Cold-start fallback recommendations.
* Interactive Streamlit user interface.
* Comprehensive exploratory data analysis.
* Custom UI themes and responsive design.

---

## Recommendation Pipeline

1. User enters technical skills.
2. Skills are converted into TF-IDF vectors.
3. Cosine Similarity calculates role relevance.
4. Job roles are ranked based on similarity scores.
5. Top matching roles are displayed.
6. Popular roles are suggested when no direct matches exist.

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* Jupyter Notebook

---

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
---

## How It Works

1. **Skill Input**
   The user enters their technical skills.
2. **Feature Extraction**
   Skills are transformed into TF-IDF vectors.
3. **Similarity Calculation**
   Cosine Similarity measures how closely user skills match each job role.
4. **Recommendation Ranking**
   Roles are ranked based on similarity scores.
5. **Personalized Suggestions**
   The Top-N most relevant technology roles are recommended.
6. **Cold-Start Handling**
   If no direct skill matches exist, the system recommends popular technology roles.

---

## Exploratory Data Analysis

The project includes detailed analysis of:

* Most demanded technical skills.
* Skill distribution across job roles.
* Certification requirements.
* Role complexity analysis.
* Description length analysis.
* Technology trends in IT careers.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/RadhikaKapoor383/SkillSync-Tech-Role-Recommendation-Engine.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the application

```bash
streamlit run App.py
```

---

## Machine Learning Techniques

* TF-IDF Vectorization
* Cosine Similarity
* Feature Engineering
* Text Preprocessing
* Content-Based Recommendation

---

## Future Improvements

* Hybrid recommendation models.
* User profile management.
* Personalized career roadmaps.
* Skill gap analysis.
* Learning resource recommendations.
* Resume-based job recommendations.

---

## Author

**Radhika Kapoor**
BS Computer Science Student
Sukkur IBA University

* GitHub: https://github.com/RadhikaKapoor383
* LinkedIn: https://www.linkedin.com/in/radhika-kapoor2005/
* Live Demo: https://skillsync-tech-role-recommendation-engine.streamlit.app/

---

## Support

If you found this project useful, please consider giving it a star on GitHub.

---

## License

This project is intended for educational, academic, and portfolio purposes.
