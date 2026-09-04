import json
import re
import io
import html
import base64
from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from fpdf import FPDF


def scrape_job_description(url: str) -> tuple[str, str]:
    if not url or not url.strip():
        return "", "No URL provided."

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url.strip(), headers=headers, timeout=12)
        response.raise_for_status()
    except requests.exceptions.MissingSchema:
        return "", "Invalid URL format. Make sure it starts with http:// or https://"
    except requests.exceptions.ConnectionError:
        return "", "Could not connect to the URL. Please check your internet or the link."
    except requests.exceptions.Timeout:
        return "", "The request timed out while fetching the job page."
    except requests.exceptions.HTTPError as e:
        return "", f"Failed to fetch page (HTTP error): {e}"
    except requests.exceptions.RequestException as e:
        return "", f"Failed to fetch the URL: {e}"

    try:
        soup = BeautifulSoup(response.content, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav",
                          "svg", "iframe", "form", "button", "input"]):
            tag.decompose()

        raw_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.splitlines()]
        lines = [line for line in lines if line]
        clean_text = "\n".join(lines)
        clean_text = re.sub(r"\n{2,}", "\n", clean_text)

        if len(clean_text) < 50:
            return "", "Page fetched, but very little readable text was found. Try pasting the JD manually."

        return clean_text, ""

    except Exception as e:
        return "", f"Failed to parse the page content: {e}"


def extract_resume_text(uploaded_file) -> tuple[str, str]:
    if uploaded_file is None:
        return "", "No resume file uploaded."

    try:
        pdf_bytes = uploaded_file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))

        if len(reader.pages) == 0:
            return "", "The uploaded PDF has no pages."

        extracted_pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)

        full_text = "\n".join(extracted_pages).strip()

        if len(full_text) < 30:
            return "", (
                "Could not extract readable text from this PDF. "
                "It might be a scanned/image-based resume (no selectable text)."
            )

        return full_text, ""

    except Exception as e:
        return "", f"Failed to read the PDF file: {e}"


ATS_PROMPT_TEMPLATE = """You are an expert Technical Recruiter and ATS (Applicant Tracking System) analyzer.

Compare the RESUME against the JOB DESCRIPTION below and evaluate the match.

JOB DESCRIPTION:
\"\"\"
{job_description}
\"\"\"

RESUME:
\"\"\"
{resume_text}
\"\"\"

Analyze skill overlap, relevant experience, and keyword alignment.

Return your response STRICTLY as raw JSON only - no markdown, no triple backticks,
no explanations outside the JSON, no leading or trailing text. The JSON object
must have EXACTLY these keys:

{{
  "match_percentage": <integer between 0 and 100>,
  "matched_skills": [<list of strings - skills/keywords present in both>],
  "missing_skills": [<list of strings - important skills from the JD missing in the resume>],
  "structural_feedback": "<string - actionable, specific tips to tailor the resume for this job>"
}}

Return ONLY the JSON object. Nothing else.
"""


def clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return text


def _sanitize_for_api(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2212": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u2022": "-",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "ignore").decode("ascii")


GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def analyze_resume_match(api_key: str, job_description: str, resume_text: str) -> tuple[dict, str]:
    if not api_key or not api_key.strip():
        return {}, "Gemini API key is missing. Please enter it in the sidebar."

    if not job_description or not job_description.strip():
        return {}, "Job description text is empty."

    if not resume_text or not resume_text.strip():
        return {}, "Resume text is empty."

    prompt = ATS_PROMPT_TEMPLATE.format(
        job_description=_sanitize_for_api(job_description[:15000]),
        resume_text=_sanitize_for_api(resume_text[:15000]),
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            params={"key": api_key.strip()},
            json=payload,
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        return {}, f"Network error while contacting Gemini: {e}"

    if response.status_code in (401, 403):
        return {}, "Invalid Gemini API key. Please check the key in the sidebar."
    if response.status_code == 404:
        return {}, f"Model '{GEMINI_MODEL}' is not available for this API key/version."
    if response.status_code == 429:
        return {}, "Gemini API quota exceeded. Please check your API usage/billing."
    if response.status_code != 200:
        return {}, f"AI analysis failed: Gemini returned HTTP {response.status_code}: {response.text[:300]}"

    try:
        data = response.json()
    except ValueError:
        return {}, "The AI response could not be parsed. Please try again."

    try:
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        finish_reason = None
        try:
            finish_reason = data["candidates"][0].get("finishReason")
        except Exception:
            pass
        if finish_reason:
            return {}, f"The AI model did not return usable content (reason: {finish_reason})."
        return {}, "The AI model returned an empty response. Please try again."

    cleaned = clean_json_response(text_out)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}, "The AI response could not be parsed as JSON. Please try again."

    result.setdefault("match_percentage", 0)
    result.setdefault("matched_skills", [])
    result.setdefault("missing_skills", [])
    result.setdefault("structural_feedback", "No feedback provided.")

    try:
        result["match_percentage"] = int(result["match_percentage"])
    except (ValueError, TypeError):
        result["match_percentage"] = 0

    if not isinstance(result["matched_skills"], list):
        result["matched_skills"] = []
    if not isinstance(result["missing_skills"], list):
        result["missing_skills"] = []
    if not isinstance(result["structural_feedback"], str):
        result["structural_feedback"] = str(result["structural_feedback"])

    return result, ""


def _sanitize_pdf_text(text) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(resume_name: str, result: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _sanitize_pdf_text("AI ATS Match Report"), ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, _sanitize_pdf_text(f"Resume: {resume_name}"), ln=True)
    pdf.cell(0, 7, _sanitize_pdf_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, _sanitize_pdf_text(f"Match Score: {result.get('match_percentage', 0)}%"), ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize_pdf_text("Matched Skills:"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    matched = result.get("matched_skills", [])
    if matched:
        for skill in matched:
            pdf.multi_cell(0, 6, _sanitize_pdf_text(f"- {skill}"))
    else:
        pdf.multi_cell(0, 6, _sanitize_pdf_text("None identified."))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize_pdf_text("Missing Skills:"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    missing = result.get("missing_skills", [])
    if missing:
        for skill in missing:
            pdf.multi_cell(0, 6, _sanitize_pdf_text(f"- {skill}"))
    else:
        pdf.multi_cell(0, 6, _sanitize_pdf_text("None identified."))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize_pdf_text("Structural Feedback:"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _sanitize_pdf_text(result.get("structural_feedback", "No feedback provided.")))

    raw_output = pdf.output()
    if isinstance(raw_output, str):
        return raw_output.encode("latin-1", "replace")
    return bytes(raw_output)


DARK_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #f0f2f6; }
    section[data-testid="stSidebar"] { background-color: #161a23; }
    section[data-testid="stSidebar"] * { color: #f0f2f6; }
    .stTextInput input, .stTextArea textarea {
        background-color: #1c1f26 !important;
        color: #f0f2f6 !important;
        border: 1px solid #333844 !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #8a8f98 !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #1c1f26 !important;
        border: 1px dashed #333844 !important;
    }
    div[data-testid="stFileUploaderDropzone"] * { color: #f0f2f6 !important; }
    .stButton>button {
        background-color: #4f8bf9;
        color: #ffffff;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover { background-color: #3d75d6; color: #ffffff; }
    .stDownloadButton>button {
        background-color: #2ecc71;
        color: #ffffff;
        border-radius: 8px;
        border: none;
    }
    div[data-testid="stMetric"] { background-color: #1c1f26; padding: 10px; border-radius: 10px; }
    div[data-testid="stMetric"] * { color: #f0f2f6 !important; }
    div[data-testid="stExpander"] { background-color: #161a23; border-radius: 8px; border: 1px solid #262b36; }
    div[data-testid="stDataFrame"] { background-color: #1c1f26; }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #f0f2f6; }
</style>
"""

LIGHT_CSS = """
<style>
    .stApp { background-color: #ffffff; color: #262730; }
    section[data-testid="stSidebar"] { background-color: #f7f8fa; }
    section[data-testid="stSidebar"] * { color: #262730; }
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d0d3d9 !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #8a8f98 !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #f7f8fa !important;
        border: 1px dashed #d0d3d9 !important;
    }
    div[data-testid="stFileUploaderDropzone"] * { color: #262730 !important; }
    .stButton>button {
        background-color: #4f8bf9;
        color: #ffffff;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover { background-color: #3d75d6; color: #ffffff; }
    .stDownloadButton>button {
        background-color: #2ecc71;
        color: #ffffff;
        border-radius: 8px;
        border: none;
    }
    div[data-testid="stMetric"] { background-color: #f7f8fa; padding: 10px; border-radius: 10px; }
    div[data-testid="stMetric"] * { color: #262730 !important; }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 8px; border: 1px solid #e6e8eb; }
    div[data-testid="stDataFrame"] { background-color: #ffffff; }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #262730; }
</style>
"""

SIDEBAR_TOOLTIP_JS = """
<script>
function setSidebarTooltips() {
    try {
        const doc = window.parent.document;
        const collapseBtn = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
        const expandBtn = doc.querySelector('[data-testid="stSidebarCollapsedControl"]');
        if (collapseBtn) { collapseBtn.title = "Collapse sidebar"; }
        if (expandBtn) { expandBtn.title = "Expand sidebar"; }
    } catch (e) {}
}
setSidebarTooltips();
try {
    const observer = new MutationObserver(setSidebarTooltips);
    observer.observe(window.parent.document.body, { childList: true, subtree: true });
} catch (e) {}
</script>
"""

THEME_TOGGLE_CSS = """
<style>
    .theme-toggle-marker { display: none; }

    div.element-container:has(div.theme-toggle-marker) + div.element-container {
        position: fixed;
        top: 14px;
        right: 24px;
        z-index: 999999;
        width: auto !important;
    }

    div.element-container:has(div.theme-toggle-marker) + div.element-container div[data-testid="stButton"] button {
        border-radius: 50% !important;
        width: 46px !important;
        height: 46px !important;
        padding: 0 !important;
        font-size: 20px !important;
        line-height: 1 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
        border: none !important;
        background-color: #4f8bf9 !important;
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
    }

    div.element-container:has(div.theme-toggle-marker) + div.element-container div[data-testid="stButton"] button:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
        background-color: #3d75d6 !important;
    }

    div[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarCollapsedControl"],
    div[data-testid="stSidebarCollapseButton"] button,
    div[data-testid="stSidebarCollapsedControl"] button {
        transition: background-color 0.15s ease-in-out, transform 0.15s ease-in-out !important;
        border-radius: 8px !important;
    }

    div[data-testid="stSidebarCollapseButton"]:hover,
    div[data-testid="stSidebarCollapsedControl"]:hover,
    div[data-testid="stSidebarCollapseButton"] button:hover,
    div[data-testid="stSidebarCollapsedControl"] button:hover {
        background-color: rgba(79, 139, 249, 0.28) !important;
        transform: scale(1.15);
        cursor: pointer;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    div[data-testid="stAlert"],
    div[data-testid="stExpander"],
    div[data-testid="stDataFrame"],
    div.ats-gauge-card,
    div.ats-hero-steps {
        animation: fadeInUp 0.45s ease-out;
    }
</style>
"""


def _render_hero_steps():
    steps_html = """
    <div class="ats-hero-steps" style="display:flex; justify-content:center; align-items:center;
         gap:22px; margin: 4px 0 26px 0; flex-wrap:wrap;">
        <div style="text-align:center; min-width:130px;">
            <div style="font-size:26px;">📋</div>
            <div style="font-weight:600; margin-top:4px; font-size:14px;">1. Paste Job</div>
            <div style="font-size:12px; opacity:0.65;">URL or JD text</div>
        </div>
        <div style="font-size:18px; opacity:0.35;">➜</div>
        <div style="text-align:center; min-width:130px;">
            <div style="font-size:26px;">📎</div>
            <div style="font-weight:600; margin-top:4px; font-size:14px;">2. Upload Resume</div>
            <div style="font-size:12px; opacity:0.65;">PDF, one or more</div>
        </div>
        <div style="font-size:18px; opacity:0.35;">➜</div>
        <div style="text-align:center; min-width:130px;">
            <div style="font-size:26px;">🎯</div>
            <div style="font-weight:600; margin-top:4px; font-size:14px;">3. Get Match Score</div>
            <div style="font-size:12px; opacity:0.65;">AI-powered ATS report</div>
        </div>
    </div>
    """
    st.markdown(steps_html, unsafe_allow_html=True)


def _render_score_gauge(match_pct: int):
    pct = max(0, min(100, match_pct))

    if pct >= 75:
        color = "#2ecc71"
    elif pct >= 45:
        color = "#f5a623"
    else:
        color = "#e74c3c"

    radius = 50
    stroke_width = 11
    circumference = 2 * 3.14159265358979 * radius
    offset = circumference * (1 - pct / 100)

    gauge_html = f"""
    <div class="ats-gauge-card" style="display:flex; flex-direction:column; align-items:center;
         justify-content:center; padding:6px 0;">
        <svg width="128" height="128" viewBox="0 0 128 128">
            <circle cx="64" cy="64" r="{radius}" fill="none" stroke="rgba(128,128,128,0.20)"
                    stroke-width="{stroke_width}"/>
            <circle cx="64" cy="64" r="{radius}" fill="none" stroke="{color}"
                    stroke-width="{stroke_width}" stroke-linecap="round"
                    stroke-dasharray="{circumference:.2f}"
                    stroke-dashoffset="{offset:.2f}"
                    transform="rotate(-90 64 64)"
                    style="transition: stroke-dashoffset 0.7s ease-in-out;"/>
            <text x="64" y="58" text-anchor="middle" font-size="24" font-weight="700"
                  fill="{color}" font-family="Helvetica, Arial, sans-serif">{pct}%</text>
            <text x="64" y="78" text-anchor="middle" font-size="10" font-weight="500"
                  fill="{color}" font-family="Helvetica, Arial, sans-serif" opacity="0.85">MATCH</text>
        </svg>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)


def _render_skill_pills(skills, kind: str = "matched"):
    if not skills:
        st.caption("None identified.")
        return

    if kind == "matched":
        bg, fg, border = "rgba(46, 204, 113, 0.15)", "#2ecc71", "rgba(46, 204, 113, 0.45)"
    else:
        bg, fg, border = "rgba(231, 76, 60, 0.15)", "#e74c3c", "rgba(231, 76, 60, 0.45)"

    pills = "".join(
        f'<span style="display:inline-block; background:{bg}; color:{fg}; '
        f'border:1px solid {border}; padding:5px 13px; margin:4px 6px 4px 0; '
        f'border-radius:16px; font-size:13px; font-weight:600;">{html.escape(str(skill))}</span>'
        for skill in skills
    )
    st.markdown(f'<div style="line-height:2.3;">{pills}</div>', unsafe_allow_html=True)


DEMO_JOB_DESCRIPTION = """Job Title: Backend Software Engineer (Python)

We are looking for a Backend Software Engineer to join our growing product team.

Responsibilities:
- Design and build RESTful APIs using Python (FastAPI or Django)
- Work with PostgreSQL and Redis for data storage and caching
- Write clean, tested code and participate in code reviews
- Deploy and monitor services on AWS using Docker and CI/CD pipelines
- Collaborate with frontend engineers consuming React-based APIs
- Optimize database queries and improve system performance

Requirements:
- 2+ years of experience with Python backend development
- Strong understanding of REST API design and SQL databases
- Experience with Docker and cloud platforms (AWS preferred)
- Familiarity with Git, CI/CD, and automated testing (pytest)
- Bonus: experience with Kubernetes, GraphQL, or Kafka
"""

DEMO_RESUME_TEXT = """Saurabh Sharma
Software Engineer

Experience:
- Built and maintained REST APIs using Python and Django for an e-commerce platform
- Used PostgreSQL for relational data storage and wrote optimized SQL queries
- Containerized services with Docker and deployed on AWS EC2
- Wrote unit and integration tests using pytest, integrated into a GitHub Actions CI pipeline
- Collaborated with a React frontend team to design and consume API contracts
- Used Git for version control in a team of 5 engineers

Skills:
Python, Django, PostgreSQL, Docker, AWS (EC2, S3), Git, pytest, REST APIs, HTML, CSS

Education:
B.Tech in Computer Science
"""

DEMO_WEAK_RESUME_TEXT = """Ananya Verma
Marketing Coordinator

Experience:
- Managed social media campaigns and content calendars for a D2C fashion brand
- Coordinated with the design team on marketing collateral and email newsletters
- Analyzed campaign performance using Google Analytics and Meta Ads Manager
- Organized influencer partnerships and brand collaborations

Skills:
Social Media Marketing, Content Strategy, Google Analytics, Canva, Email Marketing, MS Excel

Education:
BBA in Marketing
"""

DEMO_JUNIOR_RESUME_TEXT = """Rohan Mehta
Junior Software Developer

Experience:
- Assisted in building small internal tools using Python and Flask
- Wrote basic SQL queries for a college project management system using SQLite
- Familiar with Git basics through coursework
- Completed an online certification in Python programming

Skills:
Python, Flask, SQLite, basic HTML/CSS, Git (basic)

Education:
B.Sc in Computer Science (Final Year)
"""

DEMO_SCENARIOS = {
    "strong": {
        "label": "Strong Match — Senior Backend Engineer",
        "icon": "🟢",
        "job": DEMO_JOB_DESCRIPTION,
        "resumes": [{"name": "Demo_StrongMatch_Resume.pdf", "text": DEMO_RESUME_TEXT}],
    },
    "weak": {
        "label": "Weak Match — Career Switcher",
        "icon": "🟡",
        "job": DEMO_JOB_DESCRIPTION,
        "resumes": [{"name": "Demo_WeakMatch_Resume.pdf", "text": DEMO_WEAK_RESUME_TEXT}],
    },
    "multi": {
        "label": "3-Candidate Comparison",
        "icon": "👥",
        "job": DEMO_JOB_DESCRIPTION,
        "resumes": [
            {"name": "Candidate_A_Junior.pdf", "text": DEMO_JUNIOR_RESUME_TEXT},
            {"name": "Candidate_B_Senior.pdf", "text": DEMO_RESUME_TEXT},
            {"name": "Candidate_C_Mismatch.pdf", "text": DEMO_WEAK_RESUME_TEXT},
        ],
    },
}


def _add_to_history(label: str, all_results: list):
    if "history" not in st.session_state:
        st.session_state.history = []

    successful = [r for r in all_results if r["result"] is not None]
    if successful:
        top = max(successful, key=lambda x: x["result"].get("match_percentage", 0))
        if len(successful) == 1:
            summary = f"{top['result'].get('match_percentage', 0)}% match"
        else:
            summary = f"best {top['result'].get('match_percentage', 0)}% ({len(successful)} resumes)"
    else:
        summary = "failed"

    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "label": label,
        "summary": summary,
    })
    st.session_state.history = st.session_state.history[:10]


def _run_analysis(api_key: str, job_description: str, resume_items: list) -> list:
    all_results = []
    total = len(resume_items)
    progress_bar = st.progress(0.0, text="Starting analysis...")

    for idx, item in enumerate(resume_items, start=1):
        progress_bar.progress(
            (idx - 1) / total,
            text=f"Processing {item['name']} ({idx}/{total})...",
        )
        result, ai_error = analyze_resume_match(api_key, job_description, item["text"])
        if ai_error:
            all_results.append({
                "name": item["name"],
                "result": None,
                "resume_text": item["text"],
                "error": ai_error,
            })
        else:
            all_results.append({
                "name": item["name"],
                "result": result,
                "resume_text": item["text"],
                "error": "",
            })

    progress_bar.progress(1.0, text="Done!")
    progress_bar.empty()
    return all_results


def _render_single_result(resume_name: str, result: dict):
    match_pct = result.get("match_percentage", 0)
    r1, r2 = st.columns([1, 2])

    with r1:
        _render_score_gauge(match_pct)

    with r2:
        if match_pct >= 75:
            st.info("Strong match — this resume aligns well with the job.")
        elif match_pct >= 45:
            st.warning("Moderate match — some tailoring recommended.")
        else:
            st.error("Weak match — significant tailoring needed.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matched Skills**")
        _render_skill_pills(result.get("matched_skills", []), kind="matched")

    with c2:
        st.markdown("**Missing Skills**")
        _render_skill_pills(result.get("missing_skills", []), kind="missing")

    st.markdown("**Structural Feedback & Tailoring Tips**")
    st.write(result.get("structural_feedback", "No feedback provided."))

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        try:
            pdf_bytes = generate_pdf_report(resume_name, result)
            safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", resume_name.rsplit(".", 1)[0])
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"ATS_Report_{safe_filename}.pdf",
                mime="application/pdf",
                key=f"pdf_dl_{resume_name}",
            )
        except Exception as e:
            st.caption(f"(PDF report unavailable: {e})")

    with dl_col2:
        _render_copy_summary_button(resume_name, result)


def _render_copy_summary_button(resume_name: str, result: dict):
    matched = ", ".join(result.get("matched_skills", [])) or "None"
    missing = ", ".join(result.get("missing_skills", [])) or "None"
    summary_text = (
        f"{resume_name} — Match Score: {result.get('match_percentage', 0)}%\n"
        f"Matched Skills: {matched}\n"
        f"Missing Skills: {missing}"
    )
    encoded = base64.b64encode(summary_text.encode("utf-8")).decode("ascii")

    button_html = f"""
    <button onclick="
        const bytes = Uint8Array.from(atob('{encoded}'), c => c.charCodeAt(0));
        const text = new TextDecoder().decode(bytes);
        navigator.clipboard.writeText(text);
        this.innerText = 'Copied!';
        setTimeout(() => {{ this.innerText = 'Copy Summary'; }}, 1500);
    " style="
        width: 100%;
        background-color: #2ecc71;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
    ">Copy Summary</button>
    """
    st.markdown(button_html, unsafe_allow_html=True)


def get_gemini_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def main():
    st.set_page_config(
        page_title="AI Job Scraper & Resume Matcher",
        page_icon="🎯",
        layout="wide",
    )

    with st.sidebar:
        st.header("Configuration")
        api_key = get_gemini_api_key()
        if api_key:
            st.success("Gemini API key loaded from app configuration.")
        else:
            st.error(
                "Gemini API key not configured. Add GEMINI_API_KEY in "
                "Streamlit Cloud → App Settings → Secrets (or in a local "
                ".streamlit/secrets.toml file) to enable analysis."
            )

        st.markdown("---")
        st.markdown(
            "**How it works:**\n"
            "1. Paste a job URL or JD text\n"
            "2. Upload one or more resumes (PDF)\n"
            "3. Click Analyze Match\n"
        )
        st.markdown("---")
        st.markdown("**Session History**")
        history = st.session_state.get("history", [])
        if history:
            for h in history:
                st.caption(f"🕐 {h['time']} · {h['label']} — {h['summary']}")
            if st.button("Clear History", use_container_width=True, key="clear_history_btn"):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("No analyses yet this session.")
        st.markdown("---")
        st.caption("Built with Streamlit · Gemini 3.6 Flash")

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    st.markdown('<div class="theme-toggle-marker"></div>', unsafe_allow_html=True)
    toggle_icon = "☀️" if st.session_state.dark_mode else "🌙"
    if st.button(toggle_icon, key="theme_toggle_btn", help="Switch light / dark mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    dark_mode = st.session_state.dark_mode
    st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)
    st.markdown(THEME_TOGGLE_CSS, unsafe_allow_html=True)
    components.html(SIDEBAR_TOOLTIP_JS, height=0, width=0)

    st.title("AI Job Scraper & Resume Matcher")
    st.caption("Paste a job link (or JD text), upload one or more resumes, and get instant ATS-style match reports powered by Gemini.")

    _render_hero_steps()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Description")
        job_url = st.text_input("Job URL", placeholder="https://company.com/careers/job-id")
        job_text_fallback = st.text_area(
            "...or paste Job Description text directly (used if URL is empty or scraping fails)",
            height=220,
            placeholder="Paste the full job description text here as a fallback...",
        )

    with col2:
        st.subheader("Resume Upload")
        resume_files = st.file_uploader(
            "Upload Resume(s) — PDF only (upload multiple to compare candidates)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if resume_files:
            st.success(f"Uploaded {len(resume_files)} resume(s): " + ", ".join(f.name for f in resume_files))

    st.markdown("---")

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        analyze_clicked = st.button("Analyze Match", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🚀 Try Demo", use_container_width=True, help="Preview the app with built-in sample scenarios"):
            st.session_state.show_demo_picker = not st.session_state.get("show_demo_picker", False)

    demo_scenario_key = None
    if st.session_state.get("show_demo_picker", False):
        st.markdown("###### Pick a demo scenario")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            if st.button("🟢 Strong Match", use_container_width=True, key="demo_pick_strong"):
                demo_scenario_key = "strong"
        with d2:
            if st.button("🟡 Weak Match", use_container_width=True, key="demo_pick_weak"):
                demo_scenario_key = "weak"
        with d3:
            if st.button("👥 3-Candidate Compare", use_container_width=True, key="demo_pick_multi"):
                demo_scenario_key = "multi"
        with d4:
            if st.button("🔴 Failure Case", use_container_width=True, key="demo_pick_fail"):
                demo_scenario_key = "fail"
        st.markdown("---")

    def _display_results(all_results: list, job_description: str):
        successful = [r for r in all_results if r["result"] is not None]
        failed = [r for r in all_results if r["result"] is None]

        if not successful:
            st.error("Analysis failed for all uploaded resumes. See details below.")
            for r in failed:
                st.error(f"{r['name']}: {r['error']}")
            return

        st.success(f"Analysis complete for {len(successful)}/{len(all_results)} resume(s).")

        if len(all_results) > 1:
            st.markdown("## Candidate Comparison")
            comparison_rows = []
            for r in sorted(successful, key=lambda x: x["result"].get("match_percentage", 0), reverse=True):
                comparison_rows.append({
                    "Resume": r["name"],
                    "Match %": r["result"].get("match_percentage", 0),
                    "Matched Skills": len(r["result"].get("matched_skills", [])),
                    "Missing Skills": len(r["result"].get("missing_skills", [])),
                })
            for r in failed:
                comparison_rows.append({
                    "Resume": r["name"],
                    "Match %": "Error",
                    "Matched Skills": "-",
                    "Missing Skills": "-",
                })
            st.dataframe(comparison_rows, use_container_width=True, hide_index=True)

            top_candidate = max(successful, key=lambda x: x["result"].get("match_percentage", 0))
            st.info(f"Best match: {top_candidate['name']} at {top_candidate['result'].get('match_percentage', 0)}%")

            st.markdown("---")

        st.markdown("## Detailed Reports")
        for r in successful:
            with st.expander(f"{r['name']} — {r['result'].get('match_percentage', 0)}% match", expanded=(len(successful) == 1)):
                _render_single_result(r["name"], r["result"])
                st.markdown("---")
                if st.checkbox("Show raw extracted text (debug)", key=f"debug_{r['name']}"):
                    st.markdown("**Job Description (first 1000 chars):**")
                    st.text(job_description[:1000])
                    st.markdown("**Resume Text (first 1000 chars):**")
                    st.text(r["resume_text"][:1000])

        for r in failed:
            with st.expander(f"{r['name']} — analysis failed"):
                st.error(r["error"])

    if analyze_clicked:
        if not api_key or not api_key.strip():
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()

        if not job_url.strip() and not job_text_fallback.strip():
            st.error("Please provide either a Job URL or paste the Job Description text.")
            st.stop()

        if not resume_files:
            st.error("Please upload at least one resume PDF.")
            st.stop()

        job_description = ""
        with st.spinner("Fetching job description..."):
            if job_url.strip():
                scraped_text, scrape_error = scrape_job_description(job_url)
                if scrape_error:
                    if job_text_fallback.strip():
                        st.warning(f"Scraping issue: {scrape_error} — using pasted JD text instead.")
                        job_description = job_text_fallback.strip()
                    else:
                        st.error(f"Could not scrape job description: {scrape_error}")
                        st.info("Tip: paste the JD text manually in the fallback box and try again.")
                        st.stop()
                else:
                    job_description = scraped_text
            else:
                job_description = job_text_fallback.strip()

        resume_items = []
        extraction_failures = []
        for resume_file in resume_files:
            resume_text, resume_error = extract_resume_text(resume_file)
            if resume_error:
                extraction_failures.append({
                    "name": resume_file.name,
                    "result": None,
                    "resume_text": "",
                    "error": resume_error,
                })
            else:
                resume_items.append({"name": resume_file.name, "text": resume_text})

        analysis_results = _run_analysis(api_key, job_description, resume_items) if resume_items else []
        all_results = extraction_failures + analysis_results

        st.session_state.last_results = all_results
        st.session_state.last_job_description = job_description
        st.session_state.last_run_notice = None
        _add_to_history(f"Manual analysis ({len(all_results)} resume(s))", all_results)
        st.rerun()

    elif demo_scenario_key:
        if demo_scenario_key == "fail":
            job_description = DEMO_JOB_DESCRIPTION
            all_results = [{
                "name": "Corrupted_Resume_Scan.pdf",
                "result": None,
                "resume_text": "",
                "error": (
                    "Could not extract readable text from this PDF. "
                    "It might be a scanned/image-based resume (no selectable text)."
                ),
            }]
            notice = {
                "kind": "warning",
                "text": (
                    "Simulated scenario — this shows how the app surfaces a resume it couldn't read. "
                    "No API call is made for this one."
                ),
            }
            label = "Demo: Failure case"
        else:
            if not api_key or not api_key.strip():
                st.error("Please enter your Gemini API key in the sidebar to run this demo.")
                st.stop()

            scenario = DEMO_SCENARIOS[demo_scenario_key]
            job_description = scenario["job"]
            all_results = _run_analysis(api_key, job_description, scenario["resumes"])
            notice = {"kind": "info", "text": f"Running demo: {scenario['icon']} {scenario['label']}"}
            label = f"Demo: {scenario['label']}"

        st.session_state.last_results = all_results
        st.session_state.last_job_description = job_description
        st.session_state.last_run_notice = notice
        _add_to_history(label, all_results)
        st.rerun()

    st.markdown("---")

    if st.session_state.get("last_results"):
        notice = st.session_state.get("last_run_notice")
        if notice:
            if notice["kind"] == "warning":
                st.warning(notice["text"])
            else:
                st.info(notice["text"])
        _display_results(st.session_state.last_results, st.session_state.last_job_description)
    else:
        st.info(
            "👋 Paste a job description, upload a resume, or hit **Try Demo** above "
            "to see your first ATS match report here."
        )

    st.markdown("---")
    st.caption(
        "⚠️ Results are AI-generated by Gemini and may not be fully accurate — "
        "use them as a starting point, not a final verdict, and always double-check important details."
    )


if __name__ == "__main__":
    main()
