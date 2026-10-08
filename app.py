"""
================================================================================
LESSONFORGE — an AI teaching assistant, built by a teacher, for teachers.
--------------------------------------------------------------------------------
Streamlit app powered by a large language model (Hugging Face Inference API).
Three tools: Lesson Plan Generator, Quiz Builder, Explain-It-Simple.

SETUP (Streamlit Community Cloud — free):
  1. GitHub -> New repository, public, named "lesson-forge"
  2. Upload: app.py, requirements.txt (README.md optional)
  3. share.streamlit.io -> New app -> pick the repo, branch main, app.py
  4. App dashboard -> the app's three-dot menu -> Settings -> Secrets, add:
       HF_TOKEN = "hf_xxxx"
     (get a free read-only token at huggingface.co/settings/tokens)
  5. Wait a few minutes for the build. Done.

RUN LOCALLY:  export HF_TOKEN=hf_xxx && streamlit run app.py
================================================================================
"""
import os

import streamlit as st

try:
    from huggingface_hub import InferenceClient
    HAVE_HF = True
except ImportError:
    HAVE_HF = False

# Any instruction-tuned chat model works here. Swap freely.
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------
def get_client():
    """Build the API client from the secret token. Returns None if missing."""
    token = None
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        token = os.environ.get("HF_TOKEN")
    if not token or not HAVE_HF:
        return None
    return InferenceClient(model=MODEL_ID, token=token)


def generate(system_prompt, user_prompt, max_tokens=1600, temperature=0.7):
    """One LLM call. Returns text, or raises RuntimeError('no_token')."""
    client = get_client()
    if client is None:
        raise RuntimeError("no_token")
    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


SYSTEM_BASE = (
    "You are an expert instructional designer and veteran classroom teacher. "
    "You write practical, classroom-ready materials. Be specific and concrete: "
    "name actual activities, write real example questions, give time estimates. "
    "Never be vague. Never lecture about pedagogy — just produce the material."
)


# ----------------------------------------------------------------------------
# Prompt builders — the actual "engineering" in prompt engineering
# ----------------------------------------------------------------------------
def lesson_plan_prompt(topic, subject, grade, minutes, goals):
    return f"""Create a complete lesson plan using these exact markdown headers,
in this exact order:

## Learning Objectives
## Materials Needed
## Lesson Hook (first 5 minutes)
## Direct Instruction
## Guided Practice
## Independent Practice
## Closure
## Assessment
## Differentiation
## Homework / Extension

Details:
- Topic: {topic}
- Subject: {subject}
- Grade level: {grade}
- Class length: {minutes} minutes
- Learning goals: {goals or 'cover the topic thoroughly'}

Rules:
- Objectives use measurable verbs and start with "SWBAT" (students will be able to).
- Every activity includes a time estimate in minutes; times should sum to roughly the class length.
- Assessment includes a 5-question check for understanding WITH an answer key.
- Differentiation addresses advanced learners, struggling learners, and English learners separately."""


def quiz_prompt(topic, subject, grade, n_questions, qtype, include_key):
    key_rule = ("End with a clearly labeled '## Answer Key' section."
                if include_key else "Do NOT include an answer key.")
    return f"""Create a {n_questions}-question {qtype} quiz with these exact headers:

## Quiz: {topic}
## Questions
{"## Answer Key" if include_key else ""}

Details:
- Topic: {topic}
- Subject: {subject}
- Grade level: {grade}
- Question type: {qtype}

Rules:
- Number every question. For multiple choice, label options A–D.
- Questions progress from recall to application to analysis.
- Match the vocabulary and difficulty to the grade level. {key_rule}"""


def explain_prompt(concept, grade, relate_to):
    hook = f"Relate it to {relate_to} with at least one extended analogy." if relate_to else ""
    return f"""Explain the following concept so a {grade} student understands it:

Concept: {concept}

Rules:
- Open with a one-sentence plain-English definition.
- Use a concrete, everyday analogy. {hook}
- Then build up in 3 short steps from simple to slightly deeper.
- End with a 2-question self-check (questions only, no answers).
- Keep the whole thing under 250 words."""


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
GRADE_BANDS = ["K–2", "3–5", "6–8", "9–12", "College"]


def _token_banner():
    if get_client() is None:
        st.info(
            "⚙️ **Setup needed:** this app calls a language model through the Hugging Face "
            "Inference API (free). Add your token in the app's Settings → Secrets as "
            "`HF_TOKEN`, or set the `HF_TOKEN` environment variable locally. "
            "Get a free token at huggingface.co/settings/tokens."
        )


def _advanced_controls(prefix):
    with st.expander("⚙️ Advanced: model controls"):
        temperature = st.slider(
            "Temperature (creativity)", 0.0, 1.5, 0.7, 0.1,
            key=f"{prefix}_temperature",
            help="0 = strict and predictable, 1.5 = wild and creative. "
                 "Lesson plans usually want 0.5–0.8.",
        )
        max_tokens = st.slider("Max length", 400, 3000, 1600, 100,
                               key=f"{prefix}_max_tokens",
                               help="Caps how long the answer can be.")
    return temperature, max_tokens


def _render_result(title, text, filename):
    st.markdown("---")
    st.markdown(text)
    st.download_button("⬇️ Download as markdown", data=text,
                       file_name=filename, mime="text/markdown")


def tab_lesson_plan():
    st.subheader("📝 Lesson Plan Generator")
    topic = st.text_input("Topic", placeholder="e.g. The water cycle")
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", placeholder="e.g. Science")
        grade = st.selectbox("Grade level", GRADE_BANDS, index=2)
    with col2:
        minutes = st.slider("Class length (minutes)", 30, 120, 50, 5)
        goals = st.text_area("Learning goals (optional)",
                             placeholder="e.g. Students can label each stage of the cycle")
    temperature, max_tokens = _advanced_controls("lesson")
    if st.button("Generate lesson plan", type="primary", use_container_width=True):
        if not topic.strip() or not subject.strip():
            st.warning("Give me a topic and a subject first.")
            st.stop()
        prompt = lesson_plan_prompt(topic.strip(), subject.strip(), grade,
                                    minutes, goals.strip())
        with st.spinner("Writing your lesson plan..."):
            try:
                text = generate(SYSTEM_BASE, prompt, max_tokens=max_tokens,
                                temperature=temperature)
            except RuntimeError:
                st.error("No HF_TOKEN found — see the setup note at the top.")
                st.stop()
            except Exception as e:
                st.error(f"The model call failed: {e}")
                st.stop()
        _render_result("Lesson plan", text,
                       f"lesson-plan-{topic.strip().lower().replace(' ', '-')}.md")


def tab_quiz():
    st.subheader("❓ Quiz Builder")
    topic = st.text_input("Topic", placeholder="e.g. Fractions", key="quiz_topic")
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", placeholder="e.g. Math", key="quiz_subject")
        grade = st.selectbox("Grade level", GRADE_BANDS, index=2, key="quiz_grade")
    with col2:
        n_questions = st.slider("Number of questions", 3, 15, 8)
        qtype = st.selectbox("Question type",
                             ["multiple choice", "true/false", "short answer", "mixed"])
    include_key = st.checkbox("Include answer key", value=True)
    temperature, max_tokens = _advanced_controls("quiz")
    if st.button("Generate quiz", type="primary", use_container_width=True):
        if not topic.strip() or not subject.strip():
            st.warning("Give me a topic and a subject first.")
            st.stop()
        prompt = quiz_prompt(topic.strip(), subject.strip(), grade,
                             n_questions, qtype, include_key)
        with st.spinner("Writing your quiz..."):
            try:
                text = generate(SYSTEM_BASE, prompt, max_tokens=max_tokens,
                                temperature=temperature)
            except RuntimeError:
                st.error("No HF_TOKEN found — see the setup note at the top.")
                st.stop()
            except Exception as e:
                st.error(f"The model call failed: {e}")
                st.stop()
        _render_result("Quiz", text,
                       f"quiz-{topic.strip().lower().replace(' ', '-')}.md")


def tab_explain():
    st.subheader("💡 Explain It Simple")
    concept = st.text_area("Concept to explain",
                           placeholder="e.g. Why the sky is blue")
    col1, col2 = st.columns(2)
    with col1:
        grade = st.selectbox("Explain it for grade", GRADE_BANDS, index=2,
                             key="explain_grade")
    with col2:
        relate_to = st.text_input("Relate it to (optional)",
                                  placeholder="e.g. basketball",
                                  help="The explanation will use this in an analogy.")
    temperature, max_tokens = _advanced_controls("explain")
    if st.button("Explain it", type="primary", use_container_width=True):
        if not concept.strip():
            st.warning("Tell me what to explain first.")
            st.stop()
        prompt = explain_prompt(concept.strip(), grade, relate_to.strip())
        with st.spinner("Thinking of a good analogy..."):
            try:
                text = generate(SYSTEM_BASE, prompt, max_tokens=max_tokens,
                                temperature=temperature)
            except RuntimeError:
                st.error("No HF_TOKEN found — see the setup note at the top.")
                st.stop()
            except Exception as e:
                st.error(f"The model call failed: {e}")
                st.stop()
        _render_result("Explanation", text, "explanation.md")


def main():
    st.set_page_config(page_title="LessonForge", page_icon="📚", layout="wide")
    st.markdown("""
    <style>
    .stApp { background-color: #101418; }
    h1, h2, h3, p, div, span, label { color: #eef2f6; }
    .forge-title { font-size: 2.6rem; letter-spacing: .3rem; margin-bottom: 0; }
    .forge-sub { color: #9fb3c8; margin-top: 0; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="forge-title">📚 LESSONFORGE</p>', unsafe_allow_html=True)
    st.markdown('<p class="forge-sub">an AI teaching assistant — built by a teacher, for teachers</p>',
                unsafe_allow_html=True)
    _token_banner()

    tab1, tab2, tab3 = st.tabs(["📝 Lesson Plan", "❓ Quiz Builder", "💡 Explain It Simple"])
    with tab1:
        tab_lesson_plan()
    with tab2:
        tab_quiz()
    with tab3:
        tab_explain()

    st.markdown("---")
    st.caption("Powered by an open language model via the Hugging Face Inference API · "
               "prompts are the source code here")


if __name__ == "__main__":
    main()
