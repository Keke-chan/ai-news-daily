"""
AI Agent Daily Digest — Generator
Fetches 3-5 AI agent news articles via RSS, sends them to Gemini for
English-learner-friendly summaries, maintains a 30-day JSON archive,
and generates a polished single-page index.html with archive browsing.
"""

import os
import sys
import json
import glob
import datetime
import re
import feedparser
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://venturebeat.com/category/ai/feed/",
    "https://blog.google/technology/ai/rss/",
    "https://openai.com/blog/rss.xml",
]

# AI Agent related keywords for filtering
AI_AGENT_KEYWORDS = [
    "agent", "agentic", "autonomous", "mcp", "rag", "retrieval",
    "tool use", "function calling", "langchain", "langgraph",
    "autogen", "crew ai", "crewai", "openai", "anthropic", "claude",
    "gemini", "gpt", "copilot", "assistant", "automation",
    "llm", "large language model", "foundation model",
    "fine-tun", "prompt", "embeddings", "vector", "orchestrat",
    "reasoning", "chain of thought", "multi-agent", "workflow",
    "ai release", "ai update", "ai launch", "model context protocol",
]

MAX_ARTICLES = 5
MIN_ARTICLES = 3
MAX_ARTICLE_CHARS = 3000
ARCHIVE_DIR = "archives"
ARCHIVE_DAYS = 30
OUTPUT_FILE = "index.html"

# ---------------------------------------------------------------------------
# Step 1 — Fetch and filter AI Agent articles from RSS
# ---------------------------------------------------------------------------

def fetch_articles() -> list[dict]:
    """Fetch articles from multiple RSS feeds, filter for AI agent relevance."""
    candidates = []
    seen_titles = set()

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                body = (
                    entry.get("summary")
                    or entry.get("description")
                    or ""
                )
                if not body and entry.get("content"):
                    body = entry["content"][0].get("value", "")

                # Strip HTML tags for cleaner text
                body = re.sub(r"<[^>]+>", " ", body).strip()
                body = re.sub(r"\s+", " ", body)

                if not title or not body or len(body) < 80:
                    continue

                # Deduplicate by normalized title
                norm_title = title.lower().strip()
                if norm_title in seen_titles:
                    continue
                seen_titles.add(norm_title)

                # Score relevance to AI agents
                combined = (title + " " + body).lower()
                score = sum(1 for kw in AI_AGENT_KEYWORDS if kw in combined)

                if score >= 1:
                    candidates.append({
                        "title": title,
                        "link": link,
                        "body": body[:MAX_ARTICLE_CHARS],
                        "score": score,
                    })
        except Exception as exc:
            print(f"⚠ RSS error ({feed_url}): {exc}")
            continue

    # Sort by relevance score, take top N
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[:MAX_ARTICLES]

    # Remove score from output
    for a in selected:
        del a["score"]

    return selected


# ---------------------------------------------------------------------------
# Step 2 — Ask Gemini to produce structured digest JSON
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI technology journalist AND an English teacher. You write
for readers who are intermediate English learners interested in AI agents,
LLMs, and data science.

You will receive multiple news articles about AI. For EACH article, produce
a JSON object. Return a JSON array of objects with this exact structure:

[
  {
    "headline": "A clear, engaging headline rewritten in simpler English (max 12 words)",
    "one_liner": "A single-sentence summary of the article in very plain English (max 25 words)",
    "category": "release" | "technical" | "use-case" | "industry" | "research",
    "detailed_summary": "A 4-6 sentence summary in clear English. Explain technical terms when first used. Cover: what happened, why it matters, and what it means for the AI agent ecosystem.",
    "vocabulary": [
      {"term": "a key technical or advanced word from the article", "definition": "a clear English-only definition (1 sentence, under 20 words)", "example": "an example sentence using this word naturally"},
      {"term": "...", "definition": "...", "example": "..."},
      {"term": "...", "definition": "...", "example": "..."}
    ],
    "discussion_question": "One thought-provoking question about the topic that encourages critical thinking and English conversation practice."
  }
]

Rules:
- Return ONLY a valid JSON array. No markdown fences, no preamble.
- Each article gets exactly 3 vocabulary items.
- All definitions and examples must be in English only.
- Vocabulary should include technical AI terms when relevant.
- The detailed_summary should be educational — teach the reader something.
- Categories: "release" for new products/updates, "technical" for how-things-work,
  "use-case" for applications/case studies, "industry" for business/funding news,
  "research" for academic papers or breakthroughs.
"""


def call_gemini(articles: list[dict]) -> list[dict] | None:
    """Send articles to Gemini API; return parsed JSON array or None."""
    if not GEMINI_API_KEY:
        print("✖ GEMINI_API_KEY is not set.")
        return None

    articles_text = ""
    for i, a in enumerate(articles, 1):
        articles_text += f"\n--- Article {i} ---\n"
        articles_text += f"Title: {a['title']}\n"
        articles_text += f"Text: {a['body']}\n"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {"parts": [{"text": SYSTEM_PROMPT + "\n\n" + articles_text}]}
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "responseMimeType": "application/json",
                },
            },
            timeout=90,
        )
        resp.raise_for_status()

        result = resp.json()
        content = result["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(content)

        # Validate structure
        if not isinstance(data, list):
            print("✖ LLM returned non-array JSON")
            return None

        valid = []
        for item in data:
            if (
                isinstance(item.get("headline"), str)
                and isinstance(item.get("one_liner"), str)
                and isinstance(item.get("detailed_summary"), str)
                and isinstance(item.get("vocabulary"), list)
                and len(item.get("vocabulary", [])) >= 2
                and isinstance(item.get("discussion_question"), str)
            ):
                valid.append(item)

        if not valid:
            print("✖ No valid article summaries in LLM response")
            return None

        return valid

    except Exception as exc:
        print(f"✖ Gemini API call failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 3 — Archive management
# ---------------------------------------------------------------------------

def save_archive(date_str: str, articles: list[dict], digests: list[dict]):
    """Save today's data as a JSON archive file."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    archive_data = {
        "date": date_str,
        "articles": [],
    }

    for i, digest in enumerate(digests):
        source = articles[i] if i < len(articles) else {}
        archive_data["articles"].append({
            **digest,
            "source_url": source.get("link", ""),
            "source_title": source.get("title", ""),
        })

    filepath = os.path.join(ARCHIVE_DIR, f"{date_str}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(archive_data, f, ensure_ascii=False, indent=2)

    print(f"▸ Saved archive: {filepath}")


def load_archives() -> list[dict]:
    """Load all archive JSON files, sorted newest first."""
    archives = []
    pattern = os.path.join(ARCHIVE_DIR, "*.json")

    for filepath in sorted(glob.glob(pattern), reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                archives.append(data)
        except Exception as exc:
            print(f"⚠ Error loading {filepath}: {exc}")

    return archives


def cleanup_old_archives():
    """Remove archive files older than ARCHIVE_DAYS."""
    cutoff = datetime.date.today() - datetime.timedelta(days=ARCHIVE_DAYS)
    pattern = os.path.join(ARCHIVE_DIR, "*.json")

    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath).replace(".json", "")
        try:
            file_date = datetime.date.fromisoformat(basename)
            if file_date < cutoff:
                os.remove(filepath)
                print(f"▸ Removed old archive: {filepath}")
        except ValueError:
            continue


# ---------------------------------------------------------------------------
# Step 4 — Generate the HTML
# ---------------------------------------------------------------------------

def build_html(archives: list[dict]) -> str:
    """Generate the complete single-page application HTML."""
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")
    today_display = today.strftime("%B %d, %Y")

    # Serialize archives to JSON for the frontend
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
  --bg: #0c0c0e;
  --surface: #16161a;
  --surface-hover: #1c1c22;
  --surface-active: #222228;
  --border: #2a2a32;
  --border-light: #35353f;
  --text: #e8e6e3;
  --text-secondary: #9a9a9a;
  --text-muted: #6a6a72;
  --accent: #6ee7b7;
  --accent-dim: #34d3991a;
  --accent-mid: #34d39940;
  --release: #818cf8;
  --release-bg: #818cf815;
  --technical: #f97316;
  --technical-bg: #f9731615;
  --usecase: #06b6d4;
  --usecase-bg: #06b6d415;
  --industry: #f472b6;
  --industry-bg: #f472b615;
  --research: #a78bfa;
  --research-bg: #a78bfa15;
  --font-display: 'Instrument Serif', serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --radius: 12px;
  --radius-sm: 8px;
}}
*,*::before,*::after {{ margin:0;padding:0;box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{
  font-family: var(--font-body);
  background: var(--bg);
  color: var(--text);
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}}
::selection {{ background: var(--accent-mid); color: var(--text); }}

.wrapper {{ max-width: 720px; margin: 0 auto; padding: 0 1.25rem; }}

header {{
  padding: 3rem 0 2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}}
.header-top {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}}
.brand h1 {{
  font-family: var(--font-display);
  font-size: clamp(1.8rem, 5vw, 2.6rem);
  font-weight: 400;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: var(--text);
}}
.brand h1 span {{ color: var(--accent); }}
.brand p {{
  font-size: 0.875rem;
  color: var(--text-secondary);
  margin-top: 0.4rem;
  max-width: 360px;
}}
.date-nav {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.25rem;
}}
.date-nav button {{
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  width: 32px; height: 32px;
  border-radius: 50%;
  cursor: pointer;
  font-size: 0.9rem;
  display: flex; align-items: center; justify-content: center;
  transition: all 0.15s;
}}
.date-nav button:hover {{ background: var(--surface-hover); color: var(--text); }}
.date-nav button:disabled {{ opacity: 0.3; cursor: default; }}
.date-label {{
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--text-secondary);
  min-width: 130px;
  text-align: center;
}}

.stats {{
  display: flex;
  gap: 0.75rem;
  margin-top: 1.25rem;
  flex-wrap: wrap;
}}
.stat-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.7rem;
  border-radius: 50px;
  font-size: 0.75rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}}
.cat-release {{ background: var(--release-bg); color: var(--release); }}
.cat-technical {{ background: var(--technical-bg); color: var(--technical); }}
.cat-use-case {{ background: var(--usecase-bg); color: var(--usecase); }}
.cat-industry {{ background: var(--industry-bg); color: var(--industry); }}
.cat-research {{ background: var(--research-bg); color: var(--research); }}
.cat-dot {{
  width: 6px; height: 6px;
  border-radius: 50%;
  background: currentColor;
}}

.articles {{ display: flex; flex-direction: column; gap: 1rem; }}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  transition: border-color 0.2s;
}}
.card:hover {{ border-color: var(--border-light); }}
.card-header {{
  padding: 1.25rem 1.5rem;
  cursor: pointer;
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  user-select: none;
}}
.card-num {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-muted);
  background: var(--bg);
  width: 28px; height: 28px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
}}
.card-main {{ flex: 1; min-width: 0; }}
.card-headline {{
  font-family: var(--font-display);
  font-size: 1.2rem;
  font-weight: 400;
  line-height: 1.3;
  margin-bottom: 0.35rem;
}}
.card-oneliner {{
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.5;
}}
.card-meta {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-top: 0.6rem;
}}
.expand-icon {{
  color: var(--text-muted);
  font-size: 1rem;
  transition: transform 0.25s;
  flex-shrink: 0;
  margin-top: 2px;
}}
.card.open .expand-icon {{ transform: rotate(180deg); }}

.card-detail {{
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.35s ease;
}}
.card.open .card-detail {{ max-height: 1200px; }}
.detail-inner {{
  padding: 0 1.5rem 1.5rem;
  padding-left: calc(1.5rem + 28px + 1rem);
  border-top: 1px solid var(--border);
}}
.detail-section {{ margin-top: 1.25rem; }}
.detail-section h3 {{
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-bottom: 0.6rem;
}}
.detail-summary {{
  font-size: 0.95rem;
  color: var(--text);
  line-height: 1.75;
}}
.vocab-list {{ display: flex; flex-direction: column; gap: 0.75rem; }}
.vocab-item {{
  background: var(--bg);
  border-radius: var(--radius-sm);
  padding: 0.85rem 1rem;
}}
.vocab-term {{
  font-family: var(--font-mono);
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--accent);
}}
.vocab-def {{
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-top: 0.2rem;
}}
.vocab-example {{
  font-size: 0.82rem;
  color: var(--text-muted);
  font-style: italic;
  margin-top: 0.25rem;
}}
.discussion-box {{
  background: var(--accent-dim);
  border: 1px solid var(--accent-mid);
  border-radius: var(--radius-sm);
  padding: 1rem;
  font-size: 0.9rem;
  color: var(--accent);
  line-height: 1.6;
}}
.source-link {{
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.8rem;
  color: var(--text-muted);
  text-decoration: none;
  margin-top: 1rem;
  transition: color 0.15s;
}}
.source-link:hover {{ color: var(--accent); }}

.archive-toggle {{
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  font-family: var(--font-body);
  font-size: 0.8rem;
  padding: 0.45rem 1rem;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.15s;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin: 2rem auto 0;
}}
.archive-toggle:hover {{ background: var(--surface-hover); color: var(--text); }}
.archive-list {{
  display: none;
  margin-top: 1rem;
  padding-bottom: 2rem;
}}
.archive-list.show {{ display: block; }}
.archive-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.5rem;
}}
.archive-item {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.65rem 0.8rem;
  cursor: pointer;
  transition: all 0.15s;
  text-align: center;
}}
.archive-item:hover {{ background: var(--surface-hover); border-color: var(--border-light); }}
.archive-item.active {{ border-color: var(--accent); background: var(--accent-dim); }}
.archive-date {{
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text-secondary);
}}
.archive-count {{
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 0.15rem;
}}

.empty-state {{
  text-align: center;
  padding: 4rem 1rem;
  color: var(--text-muted);
}}
.empty-state h2 {{
  font-family: var(--font-display);
  font-size: 1.4rem;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
}}

footer {{
  border-top: 1px solid var(--border);
  margin-top: 3rem;
  padding: 1.5rem 0 2rem;
  text-align: center;
  font-size: 0.75rem;
  color: var(--text-muted);
}}

@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
.card {{
  animation: fadeUp 0.4s ease both;
}}
.card:nth-child(1) {{ animation-delay: 0.05s; }}
.card:nth-child(2) {{ animation-delay: 0.1s; }}
.card:nth-child(3) {{ animation-delay: 0.15s; }}
.card:nth-child(4) {{ animation-delay: 0.2s; }}
.card:nth-child(5) {{ animation-delay: 0.25s; }}

@media (max-width: 540px) {{
  header {{ padding: 2rem 0 1.5rem; }}
  .detail-inner {{ padding-left: 1.25rem; padding-right: 1.25rem; }}
  .card-header {{ padding: 1rem 1.15rem; }}
  .card-num {{ display: none; }}
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

    <button class="archive-toggle" id="archive-toggle" onclick="toggleArchive()">
      <span>&#9783;</span> Browse archive
    </button>
    <div class="archive-list" id="archive-list">
      <div class="archive-grid" id="archive-grid"></div>
    </div>
  </main>

  <footer>
    AI Agent Daily Digest &mdash; auto-generated with Gemini &amp; GitHub Actions &mdash; updated daily
  </footer>
</div>

<script>
const ARCHIVES = {archives_json};

let currentIndex = 0;

const categoryMeta = {{
  'release': {{ label: 'Release', cls: 'cat-release' }},
  'technical': {{ label: 'Technical', cls: 'cat-technical' }},
  'use-case': {{ label: 'Use case', cls: 'cat-use-case' }},
  'industry': {{ label: 'Industry', cls: 'cat-industry' }},
  'research': {{ label: 'Research', cls: 'cat-research' }},
}};

function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}}

function renderDay(index) {{
  currentIndex = index;
  const data = ARCHIVES[index];
  const container = document.getElementById('articles-container');
  const statsBar = document.getElementById('stats-bar');
  const dateLabel = document.getElementById('date-label');

  document.getElementById('prev-btn').disabled = index >= ARCHIVES.length - 1;
  document.getElementById('next-btn').disabled = index <= 0;

  if (!data || !data.articles || data.articles.length === 0) {{
    container.innerHTML = '<div class="empty-state"><h2>No articles today</h2><p>Check back tomorrow for fresh AI agent news.</p></div>';
    statsBar.innerHTML = '';
    dateLabel.textContent = data ? data.date : 'No data';
    return;
  }}

  try {{
    const d = new Date(data.date + 'T00:00:00');
    dateLabel.textContent = d.toLocaleDateString('en-US', {{ year: 'numeric', month: 'long', day: 'numeric' }});
  }} catch(e) {{
    dateLabel.textContent = data.date;
  }}

  const cats = {{}};
  data.articles.forEach(a => {{
    const c = a.category || 'technical';
    cats[c] = (cats[c] || 0) + 1;
  }});
  statsBar.innerHTML = `<span class="stat-pill" style="background:var(--accent-dim);color:var(--accent);">${{data.articles.length}} article${{data.articles.length > 1 ? 's' : ''}}</span>` +
    Object.entries(cats).map(([cat, count]) => {{
      const m = categoryMeta[cat] || categoryMeta['technical'];
      return `<span class="stat-pill ${{m.cls}}"><span class="cat-dot"></span>${{m.label}} ${{count}}</span>`;
    }}).join('');

  container.innerHTML = data.articles.map((article, i) => {{
    const cat = article.category || 'technical';
    const m = categoryMeta[cat] || categoryMeta['technical'];
    const vocabHtml = (article.vocabulary || []).map(v =>
      `<div class="vocab-item">
        <div class="vocab-term">${{esc(v.term)}}</div>
        <div class="vocab-def">${{esc(v.definition)}}</div>
        ${{v.example ? `<div class="vocab-example">"${{esc(v.example)}}"</div>` : ''}}
      </div>`
    ).join('');

    return `<div class="card" id="card-${{i}}">
      <div class="card-header" onclick="toggleCard(${{i}})">
        <span class="card-num">${{String(i+1).padStart(2,'0')}}</span>
        <div class="card-main">
          <div class="card-headline">${{esc(article.headline)}}</div>
          <div class="card-oneliner">${{esc(article.one_liner)}}</div>
          <div class="card-meta">
            <span class="stat-pill ${{m.cls}}" style="font-size:0.7rem;padding:0.2rem 0.55rem;"><span class="cat-dot"></span>${{m.label}}</span>
          </div>
        </div>
        <span class="expand-icon">&#9662;</span>
      </div>
      <div class="card-detail">
        <div class="detail-inner">
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
          ${{article.source_url ? `<a class="source-link" href="${{esc(article.source_url)}}" target="_blank" rel="noopener">Read original article &#8599;</a>` : ''}}
        </div>
      </div>
    </div>`;
  }}).join('');

  renderArchiveGrid();
}}

function toggleCard(i) {{
  const card = document.getElementById('card-' + i);
  if (card) card.classList.toggle('open');
}}

function navDate(dir) {{
  const next = currentIndex - dir;
  if (next >= 0 && next < ARCHIVES.length) renderDay(next);
}}

function toggleArchive() {{
  const list = document.getElementById('archive-list');
  list.classList.toggle('show');
}}

function renderArchiveGrid() {{
  const grid = document.getElementById('archive-grid');
  grid.innerHTML = ARCHIVES.map((a, i) => {{
    const count = a.articles ? a.articles.length : 0;
    return `<div class="archive-item ${{i === currentIndex ? 'active' : ''}}" onclick="renderDay(${{i}});window.scrollTo(0,0);">
      <div class="archive-date">${{a.date}}</div>
      <div class="archive-count">${{count}} article${{count !== 1 ? 's' : ''}}</div>
    </div>`;
  }}).join('');
}}

if (ARCHIVES.length > 0) {{
  renderDay(0);
}} else {{
  document.getElementById('articles-container').innerHTML =
    '<div class="empty-state"><h2>Welcome</h2><p>The first digest will appear after the next scheduled run.</p></div>';
}}
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
body { font-family: system-ui,sans-serif; display:flex; align-items:center; justify-content:center;
  min-height:100vh; margin:0; background:#0c0c0e; color:#e8e6e3; text-align:center; padding:2rem; }
h1 { font-size:1.4rem; margin-bottom:0.5rem; }
p { color:#9a9a9a; }
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
# Main
# ---------------------------------------------------------------------------

def main():
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    print("▸ Fetching RSS feeds…")
    articles = fetch_articles()

    if len(articles) < MIN_ARTICLES:
        print(f"✖ Only found {len(articles)} relevant articles (need {MIN_ARTICLES}+).")
        if not articles:
            html = build_fallback_html()
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write(html)
            return

    print(f"▸ Found {len(articles)} articles")
    for a in articles:
        print(f"   • {a['title'][:80]}")

    print("▸ Calling Gemini…")
    digests = call_gemini(articles)

    if not digests:
        print("✖ Gemini did not return valid data.")
        html = build_fallback_html()
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(html)
        return

    print(f"▸ Got {len(digests)} summaries")

    save_archive(today_str, articles, digests)
    cleanup_old_archives()
    archives = load_archives()

    print("▸ Building HTML…")
    html = build_html(archives)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✔ Done — wrote {OUTPUT_FILE} with {len(archives)} days of archives")


if __name__ == "__main__":
    main()
