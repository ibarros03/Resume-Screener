import streamlit as st

from utils import (
    extract_text,
    generate_ai_feedback,
    load_model,
    rank_candidates,
)

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading AI model (first time only)...")
def get_model():
    return load_model()


st.title("📄 AI Resume Screener")
st.caption("Candidate ranking based on semantic similarity, skills and generative AI.")

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    weight = st.slider(
        "Semantic similarity weight",
        0.0, 1.0, 0.6, 0.05,
        help="The remaining weight goes to the job's skill coverage.",
    )
    api_key = st.text_input(
        "Groq API Key (optional)", type="password",
        help="Without a key, feedback is generated with simple rules.",
    )
    st.markdown("---")
    st.caption(
        "⚠️ This tool supports the recruiter. The final decision should "
        "always be made by a human, since AI models can be biased."
    )

# ----------------------------------------------------------------- Inputs
col1, col2 = st.columns(2)
with col1:
    pdfs = st.file_uploader("Resumes (PDF)", type="pdf", accept_multiple_files=True)
with col2:
    jd_file = st.file_uploader("Job description (PDF or TXT)", type=["pdf", "txt"])
    jd_text = st.text_area("...or paste the job description here", height=150)

# --------------------------------------------------------------- Analysis
if st.button("🚀 Analyze candidates", type="primary"):
    # Job description text
    jd = jd_text.strip()
    if jd_file is not None:
        if jd_file.name.lower().endswith(".pdf"):
            jd = extract_text(jd_file)
        else:
            jd = jd_file.read().decode("utf-8", errors="ignore")

    if not pdfs or not jd:
        st.warning("Upload at least one resume and provide the job description.")
    else:
        resumes = {}
        for f in pdfs:
            try:
                text = extract_text(f)
            except Exception as e:
                st.error(f"Error reading {f.name}: {e}")
                continue
            if len(text) < 50:
                st.warning(f"{f.name}: no extractable text (scanned PDF?). Skipped.")
                continue
            resumes[f.name] = text

        if resumes:
            with st.spinner("Analyzing resumes..."):
                df = rank_candidates(get_model(), resumes, jd, weight)
            st.session_state["df"] = df
            st.session_state["resumes"] = resumes
            st.session_state["jd"] = jd

# ---------------------------------------------------------------- Results
if "df" in st.session_state:
    df = st.session_state["df"]
    resumes = st.session_state["resumes"]
    jd = st.session_state["jd"]

    st.subheader("🏆 Ranking")
    st.dataframe(
        df.drop(columns=["File"]),
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(df.set_index("Candidate")["Final Score (%)"])

    st.download_button(
        "⬇️ Download results (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="candidate_ranking.csv",
        mime="text/csv",
    )

    st.subheader("🧠 Individual analysis")
    for _, row in df.iterrows():
        title = f"#{row['Rank']} {row['Candidate']} — {row['Final Score (%)']}%"
        with st.expander(title):
            c1, c2, c3 = st.columns(3)
            c1.metric("Semantic", f"{row['Semantic (%)']}%")
            c2.metric("TF-IDF (baseline)", f"{row['TF-IDF (%)']}%")
            cov = row["Skill Coverage (%)"]
            c3.metric("Skill coverage", f"{cov}%" if cov == cov and cov is not None else "n/a")

            st.write(f"✅ **Matched:** {row['Matched Skills'] or '—'}")
            st.write(f"❌ **Missing:** {row['Missing Skills'] or '—'}")

            if st.button("Generate feedback", key=f"fb_{row['File']}"):
                with st.spinner("Generating feedback..."):
                    st.markdown(
                        generate_ai_feedback(row, resumes[row["File"]], jd, api_key)
                    )
                    