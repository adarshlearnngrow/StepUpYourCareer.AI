import streamlit as st
import json
import pandas as pd
import re
import time
from PyPDF2 import PdfReader
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import joblib
import numpy as np

# === Set up client ===
st.set_page_config(page_title="StepUpYourCareer.AI", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


if 'page' not in st.session_state:
    st.session_state.page = 1
if 'name' not in st.session_state:
    st.session_state.name = ""
if 'email' not in st.session_state:
    st.session_state.email = ""

def page_1():
    st.title("Welcome to StepUpYourCareer.AI")
    st.markdown("##### Let's get started with a few details.")

    name = st.text_input("Your Full Name")
    email = st.text_input("Your Email Address")

    if name and email:
        if st.button("➡️ Proceed to Resume Analysis"):
            st.session_state.name = name
            st.session_state.email = email
            st.session_state.page = 2

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

# Action Plan Generator
def extract_json_from_response(raw):
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return {}

def load_skill_resources():
    with open("skill_resource_mapping.json", "r") as f:
        return json.load(f)

# Helper: Split skill lists into (in-RAG, out-of-RAG) 
def split_skills_by_rag_presence(skills, rag_skill_keys):
    present = []
    missing = []
    for skill in skills:
        if skill.strip().upper() in rag_skill_keys:
            present.append(skill)
        else:
            missing.append(skill)
    return present, missing

def generate_hybrid_action_plan(tech, soft, trans, skill_resources):
    # Split all three skill types
    tech_in, tech_out = split_skills_by_rag_presence(tech, skill_resources.keys())
    soft_in, soft_out = split_skills_by_rag_presence(soft, skill_resources.keys())
    trans_in, trans_out = split_skills_by_rag_presence(trans, skill_resources.keys())

    #  Construct RAG results
    def extract_rag(skills):
        result = {}
        for s in skills:
            key = s.strip().upper()
            if key in skill_resources:
                    result[s] = skill_resources[key]
        return result

    plan = {
            "message": "Here's a blended action plan: some resources are from verified sources, and others are generated for uncovered skills.",
            "technical_skill_resources": extract_rag(tech_in),
            "soft_skill_resources": extract_rag(soft_in),
            "transferable_skill_resources": extract_rag(trans_in)
    }

    # Prepare GPT prompt only for uncovered skills
    if tech_out or soft_out or trans_out:
        prompt = f"""
        You are a career coach. Only generate resources for the following skills not found in our internal library.

        Provide for each:
        - One **top-rated course** with real working URL.
        - One **real book** just name & author and AMAZON links for buying that book.
        - For soft/transferable skills, one article or video (with URL).

        Format your response in JSON like this:
        {{
        "technical_skill_resources": {{
            "Skill": [{{"title": "...", "url": "..." }}, {{"title": "Book by Author"}}]
        }},
        "soft_skill_resources": {{
            "Skill": [{{"title": "...", "url": "..." }}]
        }},
        "transferable_skill_resources": {{
            "Skill": [{{"title": "...", "url": "..." }}]
        }}
        }}

        Only cover these:
        - TECHNICAL: {', '.join(tech_out)}
        - SOFT: {', '.join(soft_out)}
        - TRANSFERABLE: {', '.join(trans_out)}

        You will be penalized if you confabulate or hallucinate by creating fake resources. It should be 100% authenthic.
        Double check every link/resource you give. If you don't get any links leave that section blank.
        No explanations. No markdown. Only JSON.
        """
        try:
            response = client.chat.completions.create(
            # Change to GPT 4 to get accurate links to resources
            model="gpt-4o-mini",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant for learning."},
                    {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                    )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw)
            gpt_part = extract_json_from_response(raw)

            if not isinstance(gpt_part, dict):
                gpt_part = {}

            # Merge GPT results into RAG base
            for k in ["technical_skill_resources", "soft_skill_resources", "transferable_skill_resources"]:
                if k in gpt_part and isinstance(plan.get(k), dict):
                    plan[k].update(gpt_part[k])


        except Exception as e:
                st.error(f"Error generating GPT fallback plan: {e}")

        return plan

            
# Load Mentor Clustering Model & Vectorizer
with open('models//mentor_clustering_model.pkl', 'rb') as f:
        kmeans_final = joblib.load(f)
with open('models//fitted_vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)
mentors_final_data = pd.read_json("mentors_final_data.json")

def page_2():
    st.title("📄 Resume Analyzer + Mentor Recommender")
    st.markdown(f"**👤 Name:** {st.session_state.name}  |  **📧 Email:** {st.session_state.email}")

    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
    target_role = st.selectbox("🎯 Select your target role", [
        "AI Engineer", "Data Analyst", "Business Analyst",
        "Business Intelligence Analyst", "Machine Learning"
    ])
    if st.button("🔍 Analyze Resume") and uploaded_file and target_role:
        st.info("Processing resume... Please wait.")

        resume_text = extract_text_from_pdf(uploaded_file)
        anonymized_resume = anonymize(resume_text)

        examples, example_embeddings = load_examples()
        skills_dict = load_role_skills()
        fallback_skills = skills_dict.get(target_role, {})
        query = f"Resume: {anonymized_resume} | Role: {target_role}"
        retrieved = retrieve_examples(query, example_embeddings, examples)

        gaps = generate_skill_gap(anonymized_resume, target_role, retrieved, fallback_skills)

        st.success("Analysis complete!")

        st.subheader("🔍 Skill Gap Analysis Results")
        with st.expander("📄 Anonymized Resume"):
            st.markdown(anonymized_resume)

        st.markdown("**🎯 Target Role**")
        st.success(target_role)

        st.markdown("***Technical Skill Gaps***")
        st.markdown("\n".join(f"- {s}" for s in gaps["technical_skill_gaps"]))

        st.markdown("***Soft Skill Gaps***")
        st.markdown("\n".join(f"- {s}" for s in gaps["soft_skill_gaps"]))

        st.markdown("***Transferable Skills***")
        st.markdown("\n".join(f"- {s}" for s in gaps["transferable_skills"]))

        st.subheader("📘 Personalized Action Plan")
        tech = gaps.get("technical_skill_gaps", [])
        soft = gaps.get("soft_skill_gaps", [])
        trans = gaps.get("transferable_skills", [])
        skill_resources = load_skill_resources()
        plan = generate_hybrid_action_plan(tech, soft, trans, skill_resources)

        st.info(plan.get("message", "Here are some great learning resources!"))

        def render_resources(title, resources, emoji):
            if resources:
                st.markdown(f"### {emoji} {title}")
                for skill, items in resources.items():
                    # Skill header with bullet
                    st.markdown(f"- **{skill}**", unsafe_allow_html=True)
                    # Indented sub-bullets using HTML
                    for item in items:
                        if isinstance(item, dict):
                            label = item.get("title", "")
                            url = item.get("url", "")
                            if url:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• &nbsp; <a href='{url}' target='_blank'>{label}</a>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• &nbsp; {label}", unsafe_allow_html=True)
                        elif isinstance(item, str):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• &nbsp; {item}", unsafe_allow_html=True)


        render_resources("Technical Skill Resources", plan.get("technical_skill_resources", {}), "🛠️")
        render_resources("Soft Skill Resources", plan.get("soft_skill_resources", {}), "💬")
        render_resources("Transferable Skill Resources", plan.get("transferable_skill_resources", {}), "🔄")

        st.subheader("👥 Recommended Mentors")
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

# === Page Router ===
if st.session_state.page == 1:
    page_1()
elif st.session_state.page == 2:
    page_2()
