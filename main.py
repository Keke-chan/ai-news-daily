"""
AI Agent Daily Digest — Generator (v4.0 - Multi-Agent Architecture)
Pipeline:
  RSS Fetch (Python)
    → Agent 1: Curator    — scores all candidate articles (gemini-2.5-flash-lite, 1 call)
    → Agent 2: Selector   — picks 5 with category balance (gemini-2.5-flash-lite, 1 call)
    → Agent 3: Summarizer — deep technical summary per article (gemini-2.5-flash, 5 calls)
    → Agent 4: EnglishCoach — vocab + discussion per article (gemini-2.5-flash, 5 calls)
  HTML Build (Python)

Total API calls: ~12  |  Well within Gemini free tier (15 RPM)
"""

import os
import json
import glob
import time
import datetime
import re
import feedparser
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Cost-aware model assignment:
# Lightweight routing tasks → flash-lite (cheaper/faster)
# Deep content generation  → 2.5-flash (higher quality)
MODEL_LITE  = os.environ.get("GEMINI_MODEL_LITE",  "gemini-2.5-flash-lite")
MODEL_HEAVY = os.environ.get("GEMINI_MODEL_HEAVY", "gemini-2.5-flash")

RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://venturebeat.com/category/ai/feed/",
    "https://blog.google/technology/ai/rss/",
    "https://openai.com/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://www.technologyreview.com/feed/",
]

MAX_CANDIDATES  = 20   # articles fed into Agent 1
TARGET_ARTICLES = 5    # articles selected by Agent 2
MIN_ARTICLES    = 3    # abort threshold
MAX_ARTICLE_CHARS = 3000
AGENT_SLEEP_SEC = 5    # pause between heavy agent calls to respect 15 RPM

ARCHIVE_DIR  = "archives"
ARCHIVE_DAYS = 30
OUTPUT_FILE  = "index.html"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gemini_call(model: str, prompt: str, temperature: float = 0.5) -> str | None:
    """Single Gemini API call. Returns raw text content or None on failure."""
    if not GEMINI_API_KEY:
        print("✖ GEMINI_API_KEY is not set.")
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "responseMimeType": "application/json",
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        print(f"✖ Gemini call failed ({model}): {exc}")
        return None


def parse_json(text: str | None) -> dict | list | None:
    """Parse JSON from Gemini response text."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Strip accidental markdown fences if responseMimeType was ignored
        cleaned = re.sub(r"^```[a-z]*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            print("✖ JSON parse failed")
            return None


# ---------------------------------------------------------------------------
# Step 1 — RSS Fetch (pure Python, no LLM)
# ---------------------------------------------------------------------------

def fetch_candidates() -> list[dict]:
    """
    Fetch up to MAX_CANDIDATES articles from RSS feeds.
    Light keyword pre-filter keeps obviously unrelated articles out,
    but threshold is low (score >= 1) — the real curation is Agent 1's job.
    """
    BROAD_KEYWORDS = [
        "ai", "llm", "model", "agent", "openai", "anthropic", "google",
        "gemini", "gpt", "claude", "machine learning", "deep learning",
        "neural", "automation", "language model", "chatbot", "copilot",
    ]

    candidates = []
    seen_titles: set[str] = set()

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "").strip()
                link  = entry.get("link",  "").strip()
                body  = entry.get("summary") or entry.get("description") or ""
                if not body and entry.get("content"):
                    body = entry["content"][0].get("value", "")

                body = re.sub(r"<[^>]+>", " ", body).strip()
                body = re.sub(r"\s+", " ", body)

                if not title or not body or len(body) < 80:
                    continue

                norm = title.lower().strip()
                if norm in seen_titles:
                    continue
                seen_titles.add(norm)

                combined = (title + " " + body).lower()
                if any(kw in combined for kw in BROAD_KEYWORDS):
                    candidates.append({
                        "title": title,
                        "link":  link,
                        "body":  body[:MAX_ARTICLE_CHARS],
                    })

                if len(candidates) >= MAX_CANDIDATES:
                    break
        except Exception as exc:
            print(f"⚠ RSS error ({feed_url}): {exc}")
        if len(candidates) >= MAX_CANDIDATES:
            break

    print(f"▸ RSS fetch: {len(candidates)} candidates")
    return candidates


# ---------------------------------------------------------------------------
# Agent 1 — Curator: score every candidate article
# ---------------------------------------------------------------------------

CURATOR_PROMPT = """\
You are an expert AI news curator. Your audience is Data Scientists and Software Engineers who care about AI agents, LLMs, and the broader AI ecosystem.

Score EACH of the following articles on a scale of 1–10 for how relevant, timely, and technically interesting it is for this audience. Be strict — only truly relevant AI/ML content should score above 5.

For each article, return a JSON object. Return a JSON ARRAY of objects:
[
  {
    "title": "<exact title as given>",
    "score": <integer 1-10>,
    "reason": "<one sentence: why this score>",
    "category_hint": "release" | "technical" | "use-case" | "industry" | "research"
  }
]

Rules:
- Return ONLY a valid JSON array. No markdown, no preamble.
- Score 8-10: Major model releases, breakthrough research, key agentic framework updates.
- Score 5-7: Interesting but not groundbreaking AI industry news.
- Score 1-4: Tangentially related, business fluff, or non-technical content.
- Penalise articles that are opinion pieces with no new information.

Articles:
{articles_block}
"""

def agent_curator(candidates: list[dict]) -> list[dict]:
    """Agent 1: Score all candidate articles. Returns scored list."""
    print(f"▸ Agent 1 (Curator): scoring {len(candidates)} articles…")

    articles_block = ""
    for i, a in enumerate(candidates, 1):
        articles_block += f"\n[{i}] Title: {a['title']}\nText: {a['body'][:500]}\n"

    prompt = CURATOR_PROMPT.format(articles_block=articles_block)
    raw = gemini_call(MODEL_LITE, prompt, temperature=0.2)
    scored = parse_json(raw)

    if not isinstance(scored, list):
        print("⚠ Curator returned invalid data — falling back to all candidates")
        return [{"title": a["title"], "score": 5, "reason": "fallback", "category_hint": "technical"}
                for a in candidates]

    # Attach original body + link back from candidates by title match
    candidate_map = {a["title"]: a for a in candidates}
    result = []
    for item in scored:
        original = candidate_map.get(item.get("title", ""))
        if original:
            result.append({**item, "link": original["link"], "body": original["body"]})

    print(f"   Scored {len(result)} articles")
    return result


# ---------------------------------------------------------------------------
# Agent 2 — Selector: pick 5 balanced articles
# ---------------------------------------------------------------------------

SELECTOR_PROMPT = """\
You are an AI news editor. You have a scored list of articles. Your job is to select exactly {n} articles for today's digest.

Selection criteria:
1. Prefer high scores, but DO NOT pick 5 articles of the same category.
2. Aim for category diversity: ideally a mix of "release", "technical", "research", "use-case", "industry".
3. If two articles cover the same story, pick the better one only.
4. Today's digest should feel like a well-rounded view of the AI ecosystem.

Return a JSON array of exactly {n} objects:
[
  {{
    "title": "<exact title>",
    "category": "release" | "technical" | "use-case" | "industry" | "research",
    "selection_reason": "<one sentence: why this article was chosen>"
  }}
]

Rules:
- Return ONLY a valid JSON array. No markdown, no preamble.
- Exactly {n} items — no more, no less.

Scored articles:
{scored_block}
"""

def agent_selector(scored: list[dict], n: int = TARGET_ARTICLES) -> list[dict]:
    """Agent 2: Select n balanced articles from scored list."""
    print(f"▸ Agent 2 (Selector): selecting {n} from {len(scored)} scored articles…")

    scored_block = json.dumps(
        [{"title": s["title"], "score": s.get("score", 5),
          "reason": s.get("reason", ""), "category_hint": s.get("category_hint", "")}
         for s in scored],
        ensure_ascii=False, indent=2
    )

    prompt = SELECTOR_PROMPT.format(n=n, scored_block=scored_block)
    raw = gemini_call(MODEL_LITE, prompt, temperature=0.3)
    selected_meta = parse_json(raw)

    if not isinstance(selected_meta, list):
        print("⚠ Selector returned invalid data — falling back to top-scored articles")
        top = sorted(scored, key=lambda x: x.get("score", 0), reverse=True)[:n]
        return [{**a, "category": a.get("category_hint", "technical")} for a in top]

    # Merge selected metadata back with full article bodies
    scored_map = {s["title"]: s for s in scored}
    result = []
    for item in selected_meta[:n]:
        original = scored_map.get(item.get("title", ""))
        if original:
            result.append({
                "title":            original["title"],
                "link":             original["link"],
                "body":             original["body"],
                "category":         item.get("category", "technical"),
                "selection_reason": item.get("selection_reason", ""),
            })

    print(f"   Selected {len(result)} articles")
    return result


# ---------------------------------------------------------------------------
# Agent 3 — Summarizer: deep technical summary per article
# ---------------------------------------------------------------------------

SUMMARIZER_PROMPT = """\
You are a senior AI researcher writing for an audience of Data Scientists and Software Engineers.

Produce a deep technical summary of the following article. Your summary must go beyond surface-level reporting.

Return a single JSON object (NOT an array):
{{
  "headline": "A sharp, informative headline (max 12 words, plain English)",
  "one_liner": "One sentence capturing the core news (max 25 words)",
  "detailed_summary": "4–6 sentences. Cover: (1) what exactly happened or was released, (2) the technical mechanism or architecture involved, (3) how it compares to existing approaches or benchmarks, (4) concrete implications for practitioners building with AI. Avoid vague praise like 'this is a breakthrough'. Be specific."
}}

Rules:
- Return ONLY a valid JSON object. No markdown, no preamble.
- The detailed_summary must contain at least one concrete technical detail (e.g., parameter count, benchmark score, architecture name, API change).
- Category context: {category}

Article title: {title}
Article text:
{body}
"""

def agent_summarizer(article: dict) -> dict | None:
    """Agent 3: Generate deep technical summary for one article."""
    prompt = SUMMARIZER_PROMPT.format(
        category=article.get("category", "technical"),
        title=article["title"],
        body=article["body"],
    )
    raw = gemini_call(MODEL_HEAVY, prompt, temperature=0.4)
    result = parse_json(raw)

    if not isinstance(result, dict):
        print(f"   ⚠ Summarizer failed for: {article['title'][:60]}")
        return None

    required = ("headline", "one_liner", "detailed_summary")
    if not all(isinstance(result.get(k), str) for k in required):
        print(f"   ⚠ Summarizer missing fields for: {article['title'][:60]}")
        return None

    return result


# ---------------------------------------------------------------------------
# Agent 4 — English Coach: vocabulary + discussion per article
# ---------------------------------------------------------------------------

ENGLISH_COACH_PROMPT = """\
You are an advanced English teacher specialising in technical vocabulary for non-native speakers working in Data Science and Software Engineering.

You will receive an article and its summary. Generate vocabulary items and a discussion question.

Return a single JSON object (NOT an array):
{{
  "vocabulary": [
    {{
      "term": "a specific technical or advanced English word",
      "definition": "clear English-only definition, 1 sentence, under 20 words",
      "example": "natural example sentence using this term in a tech/DS context"
    }},
    {{ ... }},
    {{ ... }}
  ],
  "discussion_question": "One thought-provoking question about the technical or ethical implications of this article. Encourage critical thinking, not just comprehension."
}}

CRITICAL RULES:
1. Return ONLY a valid JSON object. No markdown, no preamble.
2. Exactly 3 vocabulary items.
3. BANNED basic terms (do NOT use these): "AI", "LLM", "agent", "open-source", "machine learning", "data", "model", "tool", "system", "feature".
4. CEFR level: B2–C1 or domain-specific jargon (e.g., "amortise", "inference", "proprietary", "idempotent", "throughput", "heuristic", "ablation").
5. FORBIDDEN terms already used today: {used_terms_list}
   — If a term appears in this list, skip it entirely and choose a different word.
6. The discussion question must be specific to this article, not generic.

Article title: {title}
Article summary: {summary}
Article text (for vocabulary mining):
{body}
"""

def agent_english_coach(article: dict, summary: dict, used_terms: set[str]) -> dict | None:
    """Agent 4: Generate vocabulary + discussion question for one article."""
    used_terms_list = ", ".join(sorted(used_terms)) if used_terms else "none yet"

    prompt = ENGLISH_COACH_PROMPT.format(
        used_terms_list=used_terms_list,
        title=article["title"],
        summary=article.get("detailed_summary", ""),
        body=article["body"],
    )
    raw = gemini_call(MODEL_HEAVY, prompt, temperature=0.6)
    result = parse_json(raw)

    if not isinstance(result, dict):
        print(f"   ⚠ EnglishCoach failed for: {article['title'][:60]}")
        return None

    vocab = result.get("vocabulary", [])
    if not isinstance(vocab, list) or len(vocab) < 2:
        print(f"   ⚠ EnglishCoach insufficient vocab for: {article['title'][:60]}")
        return None

    # Enforce deduplication on Python side as a safety net
    deduped = []
    for v in vocab:
        term_key = v.get("term", "").lower().strip()
        if term_key and term_key not in used_terms:
            used_terms.add(term_key)
            deduped.append(v)
    result["vocabulary"] = deduped

    return result


# ---------------------------------------------------------------------------
# Step 3 — Archive management
# ---------------------------------------------------------------------------

def save_archive(date_str: str, articles: list[dict]):
    """Save today's enriched articles as a JSON archive file."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    archive_data = {"date": date_str, "articles": articles}
    filepath = os.path.join(ARCHIVE_DIR, f"{date_str}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(archive_data, f, ensure_ascii=False, indent=2)
    print(f"▸ Saved archive: {filepath}")


def load_archives() -> list[dict]:
    """Load all archive JSON files, sorted newest first."""
    archives = []
    for filepath in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.json")), reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                archives.append(json.load(f))
        except Exception as exc:
            print(f"⚠ Error loading {filepath}: {exc}")
    return archives


def cleanup_old_archives():
    """Remove archive files older than ARCHIVE_DAYS."""
    cutoff = datetime.date.today() - datetime.timedelta(days=ARCHIVE_DAYS)
    for filepath in glob.glob(os.path.join(ARCHIVE_DIR, "*.json")):
        basename = os.path.basename(filepath).replace(".json", "")
        try:
            if datetime.date.fromisoformat(basename) < cutoff:
                os.remove(filepath)
                print(f"▸ Removed old archive: {filepath}")
        except ValueError:
            continue


# ---------------------------------------------------------------------------
# Step 4 — HTML generation (unchanged from v3)
# ---------------------------------------------------------------------------

def build_html(archives: list[dict]) -> str:
    today_display = datetime.date.today().strftime("%B %d, %Y")
    archives_json = json.dumps(archives, ensure_ascii=False)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI Agent Daily Digest</title>
<meta name="description" content="Daily AI agent news digest for English learners — releases, technical deep-dives, and use cases."/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root {{
  --bg: #0c0c0e; --surface: #16161a; --surface-hover: #1c1c22;
  --border: #2a2a32; --border-light: #35353f;
  --text: #e8e6e3; --text-secondary: #9a9a9a; --text-muted: #6a6a72;
  --accent: #6ee7b7; --accent-dim: #34d3991a; --accent-mid: #34d39940;
  --release: #818cf8;  --release-bg: #818cf815;
  --technical: #f97316; --technical-bg: #f9731615;
  --usecase: #06b6d4;  --usecase-bg: #06b6d415;
  --industry: #f472b6; --industry-bg: #f472b615;
  --research: #a78bfa; --research-bg: #a78bfa15;
  --font-display: 'Instrument Serif', serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --radius: 12px; --radius-sm: 8px;
}}
*,*::before,*::after {{ margin:0; padding:0; box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ font-family:var(--font-body); background:var(--bg); color:var(--text); line-height:1.65; -webkit-font-smoothing:antialiased; }}
::selection {{ background:var(--accent-mid); color:var(--text); }}
.wrapper {{ max-width:720px; margin:0 auto; padding:0 1.25rem; }}

header {{ padding:3rem 0 2rem; border-bottom:1px solid var(--border); margin-bottom:2rem; }}
.header-top {{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; flex-wrap:wrap; }}
.brand h1 {{ font-family:var(--font-display); font-size:clamp(1.8rem,5vw,2.6rem); font-weight:400; letter-spacing:-0.02em; line-height:1.1; }}
.brand h1 span {{ color:var(--accent); }}
.brand p {{ font-size:0.875rem; color:var(--text-secondary); margin-top:0.4rem; max-width:360px; }}
.date-nav {{ display:flex; align-items:center; gap:0.5rem; margin-top:0.25rem; }}
.date-nav button {{ background:var(--surface); border:1px solid var(--border); color:var(--text-secondary); width:32px; height:32px; border-radius:50%; cursor:pointer; font-size:0.9rem; display:flex; align-items:center; justify-content:center; transition:all 0.15s; }}
.date-nav button:hover {{ background:var(--surface-hover); color:var(--text); }}
.date-nav button:disabled {{ opacity:0.3; cursor:default; }}
.date-label {{ font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary); min-width:130px; text-align:center; }}
.stats {{ display:flex; gap:0.75rem; margin-top:1.25rem; flex-wrap:wrap; }}
.stat-pill {{ display:inline-flex; align-items:center; gap:0.35rem; padding:0.3rem 0.7rem; border-radius:50px; font-size:0.75rem; font-weight:500; letter-spacing:0.02em; }}
.cat-release {{ background:var(--release-bg); color:var(--release); }}
.cat-technical {{ background:var(--technical-bg); color:var(--technical); }}
.cat-use-case {{ background:var(--usecase-bg); color:var(--usecase); }}
.cat-industry {{ background:var(--industry-bg); color:var(--industry); }}
.cat-research {{ background:var(--research-bg); color:var(--research); }}
.cat-dot {{ width:6px; height:6px; border-radius:50%; background:currentColor; }}

.articles {{ display:flex; flex-direction:column; gap:1rem; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; transition:border-color 0.2s; animation:fadeUp 0.4s ease both; }}
.card:hover {{ border-color:var(--border-light); }}
.card:nth-child(1) {{ animation-delay:0.05s; }} .card:nth-child(2) {{ animation-delay:0.1s; }}
.card:nth-child(3) {{ animation-delay:0.15s; }} .card:nth-child(4) {{ animation-delay:0.2s; }}
.card:nth-child(5) {{ animation-delay:0.25s; }}
.card-header {{ padding:1.25rem 1.5rem; cursor:pointer; display:flex; gap:1rem; align-items:flex-start; user-select:none; }}
.card-num {{ font-family:var(--font-mono); font-size:0.7rem; color:var(--text-muted); background:var(--bg); width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px; }}
.card-main {{ flex:1; min-width:0; }}
.card-headline {{ font-family:var(--font-display); font-size:1.2rem; font-weight:400; line-height:1.3; margin-bottom:0.35rem; }}
.card-oneliner {{ font-size:0.9rem; color:var(--text-secondary); line-height:1.5; }}
.card-meta {{ display:flex; align-items:center; gap:0.6rem; margin-top:0.6rem; }}
.expand-icon {{ color:var(--text-muted); font-size:1rem; transition:transform 0.25s; flex-shrink:0; margin-top:2px; }}
.card.open .expand-icon {{ transform:rotate(180deg); }}
.card-detail {{ max-height:0; overflow:hidden; transition:max-height 0.4s ease; }}
.card.open .card-detail {{ max-height:1400px; }}
.detail-inner {{ padding:0 1.5rem 1.5rem; padding-left:calc(1.5rem + 28px + 1rem); border-top:1px solid var(--border); }}
.detail-section {{ margin-top:1.25rem; }}
.detail-section h3 {{ font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); margin-bottom:0.6rem; }}
.detail-summary {{ font-size:0.95rem; color:var(--text); line-height:1.8; }}
.vocab-list {{ display:flex; flex-direction:column; gap:0.75rem; }}
.vocab-item {{ background:var(--bg); border-radius:var(--radius-sm); padding:0.85rem 1rem; }}
.vocab-term {{ font-family:var(--font-mono); font-size:0.85rem; font-weight:500; color:var(--accent); }}
.vocab-def {{ font-size:0.85rem; color:var(--text-secondary); margin-top:0.2rem; }}
.vocab-example {{ font-size:0.82rem; color:var(--text-muted); font-style:italic; margin-top:0.25rem; }}
.discussion-box {{ background:var(--accent-dim); border:1px solid var(--accent-mid); border-radius:var(--radius-sm); padding:1rem; font-size:0.9rem; color:var(--accent); line-height:1.6; }}
.source-link {{ display:inline-flex; align-items:center; gap:0.3rem; font-size:0.8rem; color:var(--text-muted); text-decoration:none; margin-top:1rem; transition:color 0.15s; }}
.source-link:hover {{ color:var(--accent); }}

.archive-toggle {{ background:var(--surface); border:1px solid var(--border); color:var(--text-secondary); font-family:var(--font-body); font-size:0.8rem; padding:0.45rem 1rem; border-radius:50px; cursor:pointer; transition:all 0.15s; display:flex; align-items:center; gap:0.4rem; margin:2rem auto 0; }}
.archive-toggle:hover {{ background:var(--surface-hover); color:var(--text); }}
.archive-list {{ display:none; margin-top:1rem; padding-bottom:2rem; }}
.archive-list.show {{ display:block; }}
.archive-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(140px,1fr)); gap:0.5rem; }}
.archive-item {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-sm); padding:0.65rem 0.8rem; cursor:pointer; transition:all 0.15s; text-align:center; }}
.archive-item:hover {{ background:var(--surface-hover); border-color:var(--border-light); }}
.archive-item.active {{ border-color:var(--accent); background:var(--accent-dim); }}
.archive-date {{ font-family:var(--font-mono); font-size:0.75rem; color:var(--text-secondary); }}
.archive-count {{ font-size:0.7rem; color:var(--text-muted); margin-top:0.15rem; }}
.empty-state {{ text-align:center; padding:4rem 1rem; color:var(--text-muted); }}
.empty-state h2 {{ font-family:var(--font-display); font-size:1.4rem; color:var(--text-secondary); margin-bottom:0.5rem; }}
footer {{ border-top:1px solid var(--border); margin-top:3rem; padding:1.5rem 0 2rem; text-align:center; font-size:0.75rem; color:var(--text-muted); }}
@keyframes fadeUp {{ from {{ opacity:0; transform:translateY(12px); }} to {{ opacity:1; transform:translateY(0); }} }}
@media (max-width:540px) {{
  header {{ padding:2rem 0 1.5rem; }}
  .detail-inner {{ padding-left:1.25rem; padding-right:1.25rem; }}
  .card-header {{ padding:1rem 1.15rem; }}
  .card-num {{ display:none; }}
}}
</style>
</head>
<body>
<div class="wrapper">
  <header>
    <div class="header-top">
      <div class="brand">
        <h1>AI Agent <span>Daily Digest</span></h1>
        <p>Sharpen your English while staying ahead of the AI agent revolution.</p>
      </div>
      <div class="date-nav">
        <button id="prev-btn" onclick="navDate(-1)" title="Previous day">&#8249;</button>
        <span class="date-label" id="date-label">{today_display}</span>
        <button id="next-btn" onclick="navDate(1)" title="Next day">&#8250;</button>
      </div>
    </div>
    <div class="stats" id="stats-bar"></div>
  </header>
  <main>
    <div class="articles" id="articles-container"></div>
    <button class="archive-toggle" onclick="toggleArchive()">&#9783; Browse archive</button>
    <div class="archive-list" id="archive-list">
      <div class="archive-grid" id="archive-grid"></div>
    </div>
  </main>
  <footer>AI Agent Daily Digest &mdash; multi-agent pipeline &bull; Gemini &bull; GitHub Actions &mdash; updated daily</footer>
</div>
<script>
const ARCHIVES = {archives_json};
let currentIndex = 0;
const categoryMeta = {{
  'release':   {{ label:'Release',   cls:'cat-release'   }},
  'technical': {{ label:'Technical', cls:'cat-technical' }},
  'use-case':  {{ label:'Use case',  cls:'cat-use-case'  }},
  'industry':  {{ label:'Industry',  cls:'cat-industry'  }},
  'research':  {{ label:'Research',  cls:'cat-research'  }},
}};
function esc(s) {{ const d=document.createElement('div'); d.textContent=s||''; return d.innerHTML; }}
function renderDay(index) {{
  currentIndex = index;
  const data = ARCHIVES[index];
  const container = document.getElementById('articles-container');
  const statsBar  = document.getElementById('stats-bar');
  const dateLabel = document.getElementById('date-label');
  document.getElementById('prev-btn').disabled = index >= ARCHIVES.length - 1;
  document.getElementById('next-btn').disabled = index <= 0;
  if (!data || !data.articles || !data.articles.length) {{
    container.innerHTML = '<div class="empty-state"><h2>No articles today</h2><p>Check back tomorrow.</p></div>';
    statsBar.innerHTML = ''; dateLabel.textContent = data ? data.date : 'No data'; return;
  }}
  try {{
    const d = new Date(data.date + 'T00:00:00');
    dateLabel.textContent = d.toLocaleDateString('en-US', {{year:'numeric',month:'long',day:'numeric'}});
  }} catch(e) {{ dateLabel.textContent = data.date; }}
  const cats = {{}};
  data.articles.forEach(a => {{ const c = a.category||'technical'; cats[c]=(cats[c]||0)+1; }});
  statsBar.innerHTML =
    `<span class="stat-pill" style="background:var(--accent-dim);color:var(--accent);">${{data.articles.length}} article${{data.articles.length>1?'s':''}}</span>` +
    Object.entries(cats).map(([cat,count]) => {{
      const m = categoryMeta[cat]||categoryMeta['technical'];
      return `<span class="stat-pill ${{m.cls}}"><span class="cat-dot"></span>${{m.label}} ${{count}}</span>`;
    }}).join('');
  container.innerHTML = data.articles.map((article, i) => {{
    const cat = article.category||'technical';
    const m   = categoryMeta[cat]||categoryMeta['technical'];
    const vocabHtml = (article.vocabulary||[]).map(v =>
      `<div class="vocab-item">
        <div class="vocab-term">${{esc(v.term)}}</div>
        <div class="vocab-def">${{esc(v.definition)}}</div>
        ${{v.example?`<div class="vocab-example">"${{esc(v.example)}}"</div>`:''}}
      </div>`).join('');
    return `<div class="card" id="card-${{i}}">
      <div class="card-header" onclick="toggleCard(${{i}})">
        <span class="card-num">${{String(i+1).padStart(2,'0')}}</span>
        <div class="card-main">
          <div class="card-headline">${{esc(article.headline)}}</div>
          <div class="card-oneliner">${{esc(article.one_liner)}}</div>
          <div class="card-meta">
            <span class="stat-pill ${{m.cls}}" style="font-size:0.7rem;padding:0.2rem 0.55rem;">
              <span class="cat-dot"></span>${{m.label}}
            </span>
          </div>
        </div>
        <span class="expand-icon">&#9662;</span>
      </div>
      <div class="card-detail"><div class="detail-inner">
        <div class="detail-section">
          <h3>Summary</h3>
          <div class="detail-summary">${{esc(article.detailed_summary)}}</div>
        </div>
        <div class="detail-section">
          <h3>Vocabulary</h3>
          <div class="vocab-list">${{vocabHtml}}</div>
        </div>
        <div class="detail-section">
          <h3>Discussion</h3>
          <div class="discussion-box">${{esc(article.discussion_question)}}</div>
        </div>
        ${{article.source_url?`<a class="source-link" href="${{esc(article.source_url)}}" target="_blank" rel="noopener">Read original article &#8599;</a>`:''}}
      </div></div>
    </div>`;
  }}).join('');
  renderArchiveGrid();
}}
function toggleCard(i) {{ const c=document.getElementById('card-'+i); if(c) c.classList.toggle('open'); }}
function navDate(dir) {{ const n=currentIndex-dir; if(n>=0&&n<ARCHIVES.length) renderDay(n); }}
function toggleArchive() {{ document.getElementById('archive-list').classList.toggle('show'); }}
function renderArchiveGrid() {{
  const grid = document.getElementById('archive-grid');
  grid.innerHTML = ARCHIVES.map((a,i) => {{
    const count = a.articles?a.articles.length:0;
    return `<div class="archive-item ${{i===currentIndex?'active':''}}" onclick="renderDay(${{i}});window.scrollTo(0,0);">
      <div class="archive-date">${{a.date}}</div>
      <div class="archive-count">${{count}} article${{count!==1?'s':''}}</div>
    </div>`;
  }}).join('');
}}
if (ARCHIVES.length > 0) {{ renderDay(0); }}
else {{ document.getElementById('articles-container').innerHTML='<div class="empty-state"><h2>Welcome</h2><p>The first digest will appear after the next scheduled run.</p></div>'; }}
renderArchiveGrid();
</script>
</body>
</html>
"""


def build_fallback_html() -> str:
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI Agent Daily Digest</title>
<style>
body {{ font-family:system-ui,sans-serif; display:flex; align-items:center; justify-content:center;
  min-height:100vh; margin:0; background:#0c0c0e; color:#e8e6e3; text-align:center; padding:2rem; }}
h1 {{ font-size:1.4rem; margin-bottom:0.5rem; }}
p {{ color:#9a9a9a; }}
</style>
</head>
<body>
<div>
  <h1>AI Agent Daily Digest</h1>
  <p style="margin-top:1rem">We had trouble generating today's digest.</p>
  <p>Check back tomorrow!</p>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    # --- Step 1: RSS Fetch ---
    candidates = fetch_candidates()
    if len(candidates) < MIN_ARTICLES:
        print(f"✖ Only {len(candidates)} candidates found. Writing fallback.")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(build_fallback_html())
        return

    # --- Agent 1: Curator ---
    scored = agent_curator(candidates)
    if len(scored) < MIN_ARTICLES:
        print("✖ Curator returned too few results. Writing fallback.")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(build_fallback_html())
        return

    time.sleep(AGENT_SLEEP_SEC)

    # --- Agent 2: Selector ---
    selected = agent_selector(scored)
    if len(selected) < MIN_ARTICLES:
        print("✖ Selector returned too few results. Writing fallback.")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(build_fallback_html())
        return

    time.sleep(AGENT_SLEEP_SEC)

    # --- Agents 3 & 4: Summarizer + English Coach (per article) ---
    used_vocab_terms: set[str] = set()
    final_articles = []

    for i, article in enumerate(selected):
        print(f"▸ Processing article {i+1}/{len(selected)}: {article['title'][:60]}…")

        # Agent 3: Summarizer
        summary = agent_summarizer(article)
        if not summary:
            print(f"   Skipping article {i+1} — summarizer failed.")
            time.sleep(AGENT_SLEEP_SEC)
            continue

        time.sleep(AGENT_SLEEP_SEC)

        # Agent 4: English Coach
        # Pass the generated summary so vocab is grounded in the actual summary text
        article_with_summary = {**article, "detailed_summary": summary["detailed_summary"]}
        coaching = agent_english_coach(article_with_summary, summary, used_vocab_terms)

        final_article = {
            "headline":           summary.get("headline", article["title"]),
            "one_liner":          summary.get("one_liner", ""),
            "detailed_summary":   summary.get("detailed_summary", ""),
            "category":           article.get("category", "technical"),
            "vocabulary":         coaching.get("vocabulary", []) if coaching else [],
            "discussion_question": coaching.get("discussion_question", "") if coaching else "",
            "source_url":         article.get("link", ""),
            "source_title":       article.get("title", ""),
        }
        final_articles.append(final_article)

        time.sleep(AGENT_SLEEP_SEC)

    if len(final_articles) < MIN_ARTICLES:
        print(f"✖ Only {len(final_articles)} articles completed the pipeline. Writing fallback.")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(build_fallback_html())
        return

    print(f"▸ Pipeline complete: {len(final_articles)} articles processed")

    # --- Archive & HTML ---
    save_archive(today_str, final_articles)
    cleanup_old_archives()
    archives = load_archives()

    print("▸ Building HTML…")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(build_html(archives))

    print(f"✔ Done — wrote {OUTPUT_FILE} with {len(archives)} days of archives")


if __name__ == "__main__":
    main()
