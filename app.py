import streamlit as st
import pickle
import re
from pypdf import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="NLP Job & Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

# Load saved models and data
@st.cache_resource
def load_artifacts():
    with open('jobs_df.pkl', 'rb') as f:
        jobs_df = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return jobs_df, tfidf, tfidf_matrix

jobs_df, tfidf, tfidf_matrix = load_artifacts()

# Header
st.title("🎯 NLP Job & Resume Matcher")
st.markdown("""
Discover your best-fitting data science roles using **NLP TF-IDF Vectorization**, 
**Cosine Similarity**, and **Domain-Anchored Empirical Calibration**.
""")

st.write("Artifacts successfully loaded into memory!")
st.write(f"Total jobs indexed in dataset: **{len(jobs_df)}**")

# Sidebar filters
st.sidebar.header("⚙️ Match Settings")

top_n = st.sidebar.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=15,
    value=5,
    step=1
)

benchmark_score = st.sidebar.slider(
    "Calibration Benchmark (Empirical Ceiling):",
    min_value=0.20,
    max_value=0.40,
    value=0.28,
    step=0.01,
    help="Anchors the 100% confidence curve based on expected corpus sparsity."
)

st.sidebar.divider()
st.sidebar.header("📍 Candidate Preferences")

# Salary filter
max_dataset_salary = int(jobs_df['avg_salary'].max())
min_salary_filter = st.sidebar.slider(
    "Minimum Salary ($K / Year):",
    min_value=0,
    max_value=max_dataset_salary,
    value=0,
    step=5,
    help="Filter out positions with disclosed average salaries below this threshold."
)

# Location filter
all_locations = sorted([loc for loc in jobs_df['Location'].dropna().unique() if loc.strip()])
selected_locations = st.sidebar.multiselect(
    "Preferred Locations:",
    options=all_locations,
    default=[],
    help="Leave empty to match across all locations nationwide."
)

# Helpers
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calibrate_score(raw_score, benchmark):
    if raw_score <= 0:
        return 0.0
    calibrated = (raw_score / benchmark) * 100
    return round(min(calibrated, 99.0), 1)

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def format_skill_name(skill):
    skill_clean = skill.replace("_", " ")
    if skill_clean.lower() in ['aws', 'sql']:
        return skill_clean.upper()
    return skill_clean.title()

# Resume input
st.subheader("📄 Provide Your Resume / Profile")

tab_upload, tab_paste = st.tabs(["📤 Upload PDF Resume", "✍️ Paste Text Directly"])

resume_text = ""

with tab_upload:
    uploaded_pdf = st.file_uploader(
        "Upload your resume file (.pdf)",
        type=["pdf"],
        help="Extracts clean text directly from your multi-page resume PDF."
    )
    if uploaded_pdf is not None:
        try:
            resume_text = extract_text_from_pdf(uploaded_pdf)
            st.success(f"Successfully extracted {len(resume_text.split())} words from '{uploaded_pdf.name}'!")
            with st.expander("Preview Extracted Text"):
                st.text(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))
        except Exception as e:
            st.error(f"Error parsing PDF file: {e}")

with tab_paste:
    pasted_text = st.text_area(
        label="Or paste your summary/skills here:",
        height=180,
        placeholder="e.g. Data analyst skilled in Python, SQL, Tableau, Pandas, and AWS...",
        key="pasted_resume_area"
    )
    # Use pasted text if no PDF was uploaded
    if not resume_text and pasted_text.strip():
        resume_text = pasted_text

match_clicked = st.button("🔍 Find Matching Jobs", type="primary")

# Matching execution
if match_clicked:
    if not resume_text.strip():
        st.warning("⚠️ Please provide a resume by either uploading a PDF or entering text in the tabs above.")
    else:
        with st.spinner("Analyzing profile and matching across 660 roles..."):
            cleaned_input = clean_text(resume_text)
            
            # Vectorize input and compute similarity
            user_vector = tfidf.transform([cleaned_input])
            raw_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
            
            # Rank jobs by score
            scored_jobs = []
            for index, score in enumerate(raw_scores):
                scored_jobs.append((index, score))
            
            sorted_jobs = sorted(scored_jobs, key=lambda x: x[1], reverse=True)
            
            # Apply salary and location filters
            filtered_matches = []
            for job_index, raw_score in sorted_jobs:
                job_row = jobs_df.iloc[job_index]
                
                # Check salary
                salary = job_row['avg_salary']
                if min_salary_filter > 0 and salary < min_salary_filter:
                    continue
                
                # Check location
                if selected_locations:
                    if job_row['Location'] not in selected_locations:
                        continue
                
                filtered_matches.append((job_index, raw_score))
                
                if len(filtered_matches) == top_n:
                    break
            
            # Check for skills in resume
            tracked_skills = ['python', 'excel', 'hadoop', 'spark', 'aws', 'tableau', 'big_data']
            user_skills_found = []
            for skill in tracked_skills:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, cleaned_input):
                    user_skills_found.append(skill)
            
            st.divider()
            st.markdown("### 🔍 Profile Analysis")
            if user_skills_found:
                skill_badges = " ".join([f"`{format_skill_name(skill)}`" for skill in user_skills_found])
                st.markdown(f"**Detected Technical Skills:** {skill_badges}")
            else:
                st.info("No primary tracked technical skills detected in the input. Results are based on general keyword and title similarity.")
            
            st.divider()
            
            if not filtered_matches:
                st.warning("⚠️ No jobs matched your profile with the current salary and location filters. Try lowering the salary threshold or adding more locations.")
            else:
                # Analytics and charts
                
                st.markdown("### 📊 Market Fit & Compensation Analytics")

                analytics_data = []
                for job_index, raw_score in filtered_matches:
                    row = jobs_df.iloc[job_index]
                    analytics_data.append({
                        "Job Title": row['Job Title'],
                        "Company": row['Company Name'],
                        "Location": row['Location'],
                        "Salary": row['avg_salary'],
                        "Match Score": calibrate_score(raw_score, benchmark_score)
                    })

                analytics_df = pd.DataFrame(analytics_data)

                valid_salaries = analytics_df[analytics_df['Salary'] > 0]['Salary']
                avg_salary_val = int(valid_salaries.mean()) if not valid_salaries.empty else 0
                max_score_val = analytics_df['Match Score'].max()

                # Metric cards
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                with kpi_col1:
                    st.metric(
                        label="Top Match Confidence",
                        value=f"{max_score_val}%"
                    )
                with kpi_col2:
                    st.metric(
                        label="Avg. Matched Salary",
                        value=f"${avg_salary_val}K / yr" if avg_salary_val > 0 else "N/A"
                    )
                with kpi_col3:
                    st.metric(
                        label="Filtered Recommendations",
                        value=len(analytics_df)
                    )

                # Salary distribution chart
                chart_df = analytics_df[analytics_df['Salary'] > 0][['Job Title', 'Salary', 'Match Score']].copy()
                
                if not chart_df.empty:
                    chart_df['Role Label'] = analytics_df['Job Title'] + " (" + analytics_df['Company'] + ")"
                    chart_df['Salary ($K)'] = chart_df['Salary']
                
                    st.markdown("##### 💼 Estimated Salary by Top Matching Roles")
                    tab_chart, tab_table = st.tabs(["📊 Bar Chart", "📋 Data Table"])

                    with tab_chart:
                        st.bar_chart(
                            data=chart_df.set_index("Role Label")["Salary ($K)"],
                            horizontal=True,
                            color="#29b5e8",
                            y_label="Annual Salary ($K)"
                        )

                    with tab_table:
                        table_view_df = chart_df[['Role Label', 'Salary ($K)', 'Match Score']].copy()
                        table_view_df.index = range(1, len(table_view_df) + 1)
                        table_view_df.index.name = "Rank"
                        
                        st.dataframe(
                            table_view_df,
                            column_config={
                                "Role Label": st.column_config.TextColumn("Position & Company"),
                                "Salary ($K)": st.column_config.NumberColumn("Salary", format="$%dK"),
                                "Match Score": st.column_config.NumberColumn("Match Confidence", format="%.1f%%")
                            },
                            use_container_width=True
                        )

                # Display match cards
                for rank, (job_index, raw_score) in enumerate(filtered_matches, start=1): 
                    job_row = jobs_df.iloc[job_index]
                    display_score = calibrate_score(raw_score, benchmark_score)
                    job_skills = job_row['skills_extracted'].split()
                    matched_skills = []
                    missing_skills = []
                    for s in job_skills:
                        if s in user_skills_found:
                            matched_skills.append(s)
                        else:
                            missing_skills.append(s)

                    with st.container(border=True):
                        col_title, col_score = st.columns([3, 1])
                        
                        with col_title:
                            st.subheader(f"#{rank} {job_row['Job Title']}")
                            st.caption(f"🏢 **{job_row['Company Name']}**  |  📍 {job_row['Location']}")
                        
                        with col_score:
                            st.metric(
                                label="Match Confidence",
                                value=f"{display_score}%",
                                delta=f"Raw: {round(raw_score, 3)}"
                            )
                        
                        salary_display = f"${job_row['avg_salary']}K / year" if job_row['avg_salary'] > 0 else "Not Disclosed"
                        st.write(f"💰 **Estimated Salary:** {salary_display}")
                        
                        if matched_skills:
                            matched_str = " ".join([f":green[✔ {format_skill_name(s)}]" for s in matched_skills])
                            st.markdown(f"**Matched Competencies:** {matched_str}")
                        else:
                            st.markdown("**Matched Competencies:** _General role & context overlap_")

                        if missing_skills:
                            missing_str = " ".join([f":red[✗ {format_skill_name(s)}]" for s in missing_skills])
                            st.markdown(f"**Skill Gap to Acquire:** {missing_str}")