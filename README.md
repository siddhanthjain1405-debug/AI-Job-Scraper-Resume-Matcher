# AI Job Scraper & Resume Matcher

An AI-powered ATS (Applicant Tracking System) simulator that compares a resume against a job description and returns a match score, matched/missing skills, and actionable resume-tailoring feedback — built as a 1-week Build Sprint MVP for The Skillians' Generative AI Developer Internship.

**Live Demo:** [add your deployed Streamlit URL here]
**Video/Screenshots:** [optional — add if you record a short walkthrough]

---

## What it does

1. Takes a job description — either a job posting URL (auto-scraped) or pasted text.
2. Takes one or more resumes (PDF upload).
3. Sends both to Google's Gemini model with a structured prompt and gets back:
   - A match percentage
   - A list of matched skills
   - A list of missing skills
   - Actionable, resume-specific tailoring feedback
4. If multiple resumes are uploaded against the same job, it ranks them in a comparison table and highlights the best-fit candidate.
5. Every report can be exported as a PDF or copied as a quick text summary.

A **Demo Mode** is built in (4 preset scenarios — strong match, weak match, multi-candidate comparison, and a simulated failure case) so the app can be evaluated instantly without needing a real job link or resume on hand.

---

## My approach

The brief asked for a lightweight, single-file MVP, so I prioritized a **simple, dependable pipeline over unnecessary complexity**:

- **Scraping over APIs:** Most job boards don't offer free public APIs, so I used `requests` + `BeautifulSoup` to fetch and clean visible page text. Since JS-heavy sites (like some LinkedIn postings) won't render via a static fetch, I added a manual "paste JD text" fallback so the app never hard-blocks the user.
- **Prompt-based structured output over fine-tuning:** Given the 1-week timeframe, I used prompt engineering to force Gemini to return **strict JSON** (`match_percentage`, `matched_skills`, `missing_skills`, `structural_feedback`) instead of freeform text, then defensively parsed and validated that JSON so a malformed AI response never crashes the app.
- **Direct REST call instead of the SDK:** I initially used Google's `google-genai` SDK, but hit a platform-specific bug where its internal HTTP layer failed to encode certain Unicode characters (common in real resumes/JDs — em dashes, smart quotes, bullets). I replaced it with a direct call to the Gemini REST endpoint via `requests`, which JSON-encodes everything ASCII-safe by default. This removed an entire class of encoding bugs and reduced a dependency.
- **Graceful degradation everywhere:** every stage (scraping, PDF parsing, AI call, JSON parsing) has its own try/except with a specific, user-facing error message — the goal was that nothing in the app should throw a raw traceback at the user.
- **Session-state persistence:** results are stored in Streamlit's `session_state` rather than only existing for one script run, so interacting with any other widget (like the debug checkbox) doesn't wipe the report — and it enables a lightweight session history in the sidebar.
- **Kept it a single `app.py`:** per the constraint of running comfortably on a 4GB RAM machine, there's no heavy local ML model — all intelligence comes from the Gemini API call, keeping the app's own footprint tiny.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| UI | [Streamlit](https://streamlit.io/) | Fast to build, no HTML/CSS boilerplate needed for a functional MVP |
| Job scraping | `requests` + `BeautifulSoup` | Lightweight HTML fetch + text cleaning, no headless browser needed |
| Resume parsing | `PyPDF2` | Simple, dependency-light PDF text extraction |
| AI engine | Google **Gemini** (`gemini-3.6-flash`) via direct REST API | Fast, cheap, strong instruction-following for structured JSON output |
| PDF report export | `fpdf2` | Lightweight PDF generation, no LaTeX/heavy renderer needed |
| Styling | Custom CSS injected via `st.markdown` | Dark/light theme, gauge, skill-pill badges — without leaving Streamlit |

---

## Project structure

```
.
├── app.py                          # Entire application (UI + scraper + resume parser + AI engine)
├── requirements.txt                 # Python dependencies
├── .streamlit/
│   └── secrets.toml.example         # Template showing the required secret key name (safe to commit)
├── .gitignore
└── README.md
```

---

## Running it locally

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

Create a local secrets file (this file is git-ignored and never committed):

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Open `.streamlit/secrets.toml` and paste your own free Gemini API key (get one at https://aistudio.google.com/app/apikey):

```toml
GEMINI_API_KEY = "your-actual-key-here"
```

Then run:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## A note on the API key

The app reads its Gemini API key from **Streamlit's built-in secrets manager** (`st.secrets["GEMINI_API_KEY"]`) — it is never hardcoded, never displayed in the UI, and never committed to the repository (`.streamlit/secrets.toml` is git-ignored; only a placeholder `.streamlit/secrets.toml.example` is committed).

- Locally, the key comes from `.streamlit/secrets.toml` on your own machine.
- On Streamlit Community Cloud, the key is set once in **App Settings → Secrets** and stays server-side.
- Evaluators testing the live deployed link do **not** need their own API key and never see the key — they just use the app directly.
- The GitHub repo and deployed app are safe to share publicly — there is no secret in the source code.

---

## Known limitations

- JavaScript-heavy job pages may not scrape cleanly (use the "paste JD text" fallback in that case).
- Scanned/image-only PDF resumes have no selectable text and can't be parsed (a clear error is shown instead of failing silently).
- Match scoring is AI-generated and should be treated as directional guidance, not a definitive verdict — this is called out in the app's footer.

---

## Author

Built by Siddhanth Jain as part of The Skillians' 1-week Generative AI Developer Internship Build Sprint.
