import streamlit as st
import json
import pandas as pd
import re
import time
from PyPDF2 import PdfReader
from openai import OpenAI
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import joblib

# === Set up client ===
client = OpenAI(api_key="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")


# === Load Examples and Embeddings ===
@st.cache_data
def load_examples():
    with open("skill_gap_analysis.json", "r") as f:
        examples = json.load(f)
    example_texts = [
        f"Resume: {ex['resume_summary']} | Role: {ex['target_role']}" for ex in examples
    ]
    embeddings = [get_embedding(text) for text in example_texts]
    return examples, embeddings

# === Load Role Skills ===
@st.cache_data
def load_role_skills():
    with open("role_skills.json", "r") as f:
        role_skills_list = json.load(f)
    return {
        entry["role"]: {
            "technical_skills": entry.get("technical_skills", []),
            "soft_skills": entry.get("soft_skills", [])
        }
        for entry in role_skills_list if "role" in entry
    }

# Embedding
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Resume text extraction ===
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

# Resume anonymizer
def anonymize(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that removes personal identifiers from resumes."},
            {"role": "user", "content": f"Anonymize this resume:\n\n{text}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# RAG-based retrieval 
def retrieve_examples(query, embeddings, examples, k=3):
    query_emb = get_embedding(query)
    sims = sk_cosine([query_emb], embeddings)[0]
    top_k = sims.argsort()[-k:][::-1]
    return [examples[i] for i in top_k]

# Skill gap analysis 
def generate_skill_gap(resume_text, target_role, retrieved_examples, fallback_skills):
    examples_prompt = "\n\n".join([
        f"Example for role {ex['target_role']}:\nResume: {ex['resume_summary']}\nSkill Gaps: tech={ex['technical_skill_gap']}, soft={ex['soft_skill_gap']}, transferable={ex['transferable_skills']}"
        for ex in retrieved_examples
    ])
    expected_tech = fallback_skills.get("technical_skills", [])
    expected_soft = fallback_skills.get("soft_skills", [])

    prompt = f"""
You are a highly experienced career advisor. Based on examples and required skills, identify skill gaps.

## EXAMPLES
{examples_prompt}

## TASK:

Analyze the new resume below and return only JSON:
{{
  "technical_skill_gaps": [list of missing technical skills],
  "soft_skill_gaps": [list of missing soft skills],
  "transferable_skills": [skills from the resume that help bridge gaps]
}}

No explanation, no markdown, just clean JSON.

Target Role: {target_role}
Required Technical Skills: {', '.join(expected_tech)}
Required Soft Skills: {', '.join(expected_soft)}
Resume:
{resume_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant for skill gap analysis."},
                  {"role": "user", "content": prompt}],
        temperature=0.0
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw)
    return json.loads(raw)

# === Action Plan Generator ===
def extract_json_from_response(raw):
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return {}

def generate_action_plan(tech, soft, trans):
    prompt = f"""
You are a career coach. For each skill, return 2–3 real learning resources:

- One top-rated **online course** with title and valid working URL from platforms like Coursera, Udemy, edX, etc.
- One **real book** with ONLY its name and author (no links).
- For soft and transferable skills, 1 real article or course.

Return only JSON in this format:

{{
  "message": "motivational message",
  "technical_skill_resources": {{
    "Skill A": [
      {{"title": "Course Title", "url": "https://..." }},
      {{"title": "Book Title by Author" }}
    ]
  }},
  "soft_skill_resources": {{
    "Skill B": [
      {{"title": "Article Title", "url": "https://..." }}
    ]
  }},
  "transferable_skill_resources": {{
    "Skill C": [
      {{"title": "YouTube Video", "url": "https://..." }}
    ]
  }}
}}

No markdown. No explanation. Only real, existing links.

Technical Skills Gap: {', '.join(tech)}
Soft Skills Gap: {', '.join(soft)}
Transferable Skills: {', '.join(trans)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for learning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw)
        return extract_json_from_response(raw)
    except Exception as e:
        st.error(f"Error generating action plan: {e}")
        return {}
    
# === Load Mentor Clustering Model & Vectorizer === 

with open('models//mentor_clustering_model.pkl', 'rb') as f:
  kmeans_final = joblib.load(f)

with open('models//fitted_vectorizer.pkl', 'rb') as f:
  vectorizer = joblib.load(f)

mentors_final_data = pd.read_json("mentors_final_data.json")


# === Streamlit App ===
def main():
    st.set_page_config("Skill Gap Analyzer & Mentor Recommender", layout="wide")
    st.title("StepUpYourCareer.AI")

    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
    # Example list — feel free to expand or load dynamically
    available_roles = [
            "AI Engineer",
            "Data Analyst",
            "Business Analyst",
            "Business Intelligence Analyst",
            "Machine Learning"
        ]

    target_role = st.selectbox("Select your target role", options=available_roles)
    submitted = st.button("Submit")

    if submitted:
        if uploaded_file and target_role:
            st.info("Processing resume... This might take ~1 min")

            resume_text = extract_text_from_pdf(uploaded_file)
            anonymized_resume = anonymize(resume_text)

            examples, example_embeddings = load_examples()
            skills_dict = load_role_skills()

            fallback_skills = skills_dict.get(target_role, {})
            query = f"Resume: {anonymized_resume} | Role: {target_role}"
            retrieved = retrieve_examples(query, example_embeddings, examples)

            gaps = generate_skill_gap(anonymized_resume, target_role, retrieved, fallback_skills)

            st.subheader("🔍 Skill Gap Analysis Results")
            with st.expander("📄 Anonymized Resume"):
                st.markdown(anonymized_resume)

            st.markdown("Target Role")
            st.success(target_role)

            st.markdown("Technical Skill Gaps")
            for skill in gaps["technical_skill_gaps"]:
                st.markdown(f"- {skill}" if skill else "No gaps!")

            st.markdown("Soft Skill Gaps")
            for skill in gaps["soft_skill_gaps"]:
                st.markdown(f"- {skill}" if skill else "No gaps!")

            st.markdown("Transferable Skills")
            for skill in gaps["transferable_skills"]:
                st.markdown(f"- {skill}" if skill else "None found.")

            st.subheader("Action Plan (Learning Resources)")
            tech = gaps.get("technical_skill_gaps", [])
            soft = gaps.get("soft_skill_gaps", [])
            trans = gaps.get("transferable_skills", [])

            plan = generate_action_plan(tech, soft, trans)
            st.info(plan.get("message", "Here are some great learning resources!"))

            def render_resources(title, resources, emoji):
                if resources:
                    st.markdown(f"### {emoji} {title}")
                    for skill, items in resources.items():
                        st.markdown(f"**🔹 {skill}**")
                        for item in items:
                            if isinstance(item, dict):
                                title = item.get("title", "")
                                url = item.get("url", "")
                                if url:
                                    st.markdown(f"- [{title}]({url})")
                                else:
                                    st.markdown(f"- {title}")

            render_resources("Technical Skill Resources", plan.get("technical_skill_resources", {}), "🛠️")
            render_resources("Soft Skill Resources", plan.get("soft_skill_resources", {}), "💬")
            render_resources("Transferable Skill Resources", plan.get("transferable_skill_resources", {}), "🔄")

            # === Mentor Recommendation ===
            st.subheader("Recommended Mentors")

            try:
                skill_list = [s.strip() for s in tech if s.strip()]
                mlb_vector = vectorizer.transform([skill_list])
                cluster_id = kmeans_final.predict(mlb_vector)[0]
                mentors = mentors_final_data[mentors_final_data["cluster"] == cluster_id]

                if not mentors.empty:
                    for _, row in mentors.iterrows():
                        st.markdown(f"**👨‍🏫 {row['name']}**")
                        st.markdown(f"- 🔗 [LinkedIn](https://www.linkedin.com/in/{row['linkedin_id']})")
                        st.markdown(f"- Technical Skills: {', '.join(row['technical_skills'])}")
                        st.markdown(f"- Bio: {row['bio']}")
                        st.markdown("---")
                else:
                    st.warning("No matching mentors found.")
            except Exception as e:
                st.error(f"Mentor recommendation error: {e}")

            # === Save Results Button ===
            if st.button("💾 Save Results"):
                result = {
                    "target_role": target_role,
                    "resume_text": anonymized_resume,
                    **gaps,
                    "action_plan": plan
                }
                with open("final_skill_gap_output.json", "w") as f:
                    json.dump(result, f, indent=2)
                st.success("Saved as final_skill_gap_output.json")


if __name__ == "__main__":
    main()