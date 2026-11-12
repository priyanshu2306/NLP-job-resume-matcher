# 🎯 NLP Job & Resume Matcher

An interactive Natural Language Processing (NLP) job recommendation and skill-gap analysis dashboard built with **Python**, **Streamlit**, and **Scikit-Learn**. 

Trained and indexed on a curated **Glassdoor Data Science job postings dataset** (660+ roles), this application is specifically tailored to evaluate candidate resumes against **Data Science, Data Analytics, and Machine Learning** positions.

## 🧠 How It Works
1. **Text Preprocessing & Vectorization**: Cleans raw resume text and transforms it into numerical feature vectors using a pre-fitted **TF-IDF Vectorizer** trained on industry job descriptions.
2. **Cosine Similarity Ranking**: Computes vector similarity between the candidate's profile and all indexed Glassdoor listings to rank roles by contextual and technical overlap.
3. **Domain-Anchored Calibration**: Normalizes raw sparse cosine similarity scores against an adjustable empirical ceiling so match percentages remain intuitive and interpretable.
4. **Two-Stage Retrieval**: Filters top-ranked matches by minimum disclosed salary and preferred locations while deduplicating repeat postings.

## ✨ Key Features
- **Dual Resume Ingestion**: Supports multi-page PDF resume extraction (`pypdf`) alongside direct text input.
- **Market Fit & Compensation Analytics**: Displays executive KPIs (Top Match Confidence, Average Matched Salary) alongside a horizontal compensation bar chart and a rank-indexed data table.
- **Skill-Gap Diagnostics**: Scans resumes against core data science competencies (`Python`, `SQL`, `AWS`, `Spark`, `Hadoop`, `Tableau`, `Excel`, `Big Data`) to highlight matched strengths and missing skills per role.


## Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```