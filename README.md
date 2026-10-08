# LessonForge — AI Teaching Assistant

Built by a teacher, for teachers. Three tools powered by an open language model:

- **Lesson Plan Generator** — topic + grade + class length → full lesson plan with objectives, timed activities, assessment, and differentiation
- **Quiz Builder** — topic → leveled quiz with answer key
- **Explain It Simple** — any concept explained at any grade level, with an optional personal analogy

## Deploy (Streamlit Community Cloud — free)

1. GitHub: new **public** repository named `lesson-forge`
2. Upload `app.py` and `requirements.txt` to the repo (branch `main`)
3. Go to [share.streamlit.io](https://share.streamlit.io) → New app → pick the repo, branch `main`, file `app.py`
4. In the app dashboard, open the app's three-dot menu → **Settings** → **Secrets**, and add:
   ```toml
   HF_TOKEN = "hf_your_token_here"
   ```
   Get a free read-only token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Deploy and wait a few minutes for the build

## Local run

```bash
export HF_TOKEN=hf_your_token_here
streamlit run app.py
```
