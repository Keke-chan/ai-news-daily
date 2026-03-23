"""
AI News for English Learners — Daily Generator
Fetches an AI news article via RSS, sends it to an LLM for simplification,
and generates a static index.html for GitHub Pages.
"""

import os
import sys
import json
import datetime
import feedparser
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
]

MAX_ARTICLE_CHARS = 4000  # trim long articles before sending to the LLM
OUTPUT_FILE = "index.html"

# ---------------------------------------------------------------------------
# Step 1 — Fetch the latest article from RSS
# ---------------------------------------------------------------------------

def fetch_latest_article() -> dict | None:
    """Try each RSS feed in order; return the first valid article dict."""
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:  # check first 5 entries
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                # Prefer 'summary' but fall back to 'description' or 'content'
                body = (
                    entry.get("summary")
                    or entry.get("description")
                    or ""
                )
                if not body and entry.get("content"):
                    body = entry["content"][0].get("value", "")

                if title and body and len(body) > 100:
                    return {
                        "title": title,
                        "link": link,
                        "body": body[:MAX_ARTICLE_CHARS],
                    }
        except Exception as exc:
            print(f"⚠ RSS error ({feed_url}): {exc}")
            continue

    return None


# ---------------------------------------------------------------------------
# Step 2 — Ask the LLM to produce the lesson JSON
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an English teacher who helps ESL / English-learner students understand
technology news. You will receive an article about AI or technology.

Return ONLY valid JSON (no markdown fences) with this exact structure:

{
  "summary": "A simple English summary of the article in at most 5 short sentences. Use plain language suitable for an intermediate English learner.",
  "vocabulary": [
    {"word": "example", "definition": "a simple definition in plain English"},
    {"word": "...", "definition": "..."}
  ],
  "example_sentence": "One sentence that naturally uses one of the vocabulary words.",
  "discussion_question": "One simple question an English learner could discuss with a partner."
}

Rules:
- vocabulary must contain exactly 5 items.
- Keep every definition to one sentence, under 15 words.
- The summary should avoid jargon; if a technical term is necessary, explain it.
"""


def call_llm(article: dict) -> dict | None:
    """Send article to OpenAI-compatible API; return parsed JSON or None."""
    if not OPENAI_API_KEY:
        print("✖ OPENAI_API_KEY is not set.")
        return None

    user_message = (
        f"Article title: {article['title']}\n\n"
        f"Article text:\n{article['body']}"
    )

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)

        # Basic validation
        assert isinstance(data.get("summary"), str), "missing summary"
        assert len(data.get("vocabulary", [])) == 5, "need 5 vocab items"
        assert isinstance(data.get("example_sentence"), str), "missing example"
        assert isinstance(data.get("discussion_question"), str), "missing question"

        return data

    except Exception as exc:
        print(f"✖ LLM call failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 3 — Generate the HTML page
# ---------------------------------------------------------------------------

def build_html(article: dict, lesson: dict) -> str:
    """Return a complete, self-contained HTML string."""
    today = datetime.date.today().strftime("%B %d, %Y")

    vocab_html = ""
    for item in lesson["vocabulary"]:
        word = _esc(item["word"])
        defn = _esc(item["definition"])
        vocab_html += (
            f'<li><span class="vocab-word">{word}</span>'
            f' — <span class="vocab-def">{defn}</span></li>\n'
        )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI News for English Learners</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
:root {{
  --bg: #f8f5f0;
  --card: #ffffff;
  --text: #2c2c2c;
  --muted: #6b6b6b;
  --accent: #e05a2b;
  --accent-light: #fff1ec;
  --border: #e4ddd5;
  --radius: 12px;
}}
*, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'IBM Plex Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  padding: 2rem 1rem 4rem;
}}
.container {{ max-width: 620px; margin: 0 auto; }}
header {{
  text-align: center;
  margin-bottom: 2.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 2px solid var(--border);
}}
header h1 {{
  font-family: 'DM Serif Display', serif;
  font-size: clamp(1.6rem, 5vw, 2.2rem);
  letter-spacing: -0.02em;
  color: var(--text);
}}
header p {{
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: 0.3rem;
}}
.date-badge {{
  display: inline-block;
  background: var(--accent);
  color: #fff;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.25rem 0.8rem;
  border-radius: 50px;
  margin-bottom: 0.8rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
section {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  margin-bottom: 1.25rem;
}}
section h2 {{
  font-family: 'DM Serif Display', serif;
  font-size: 1.15rem;
  margin-bottom: 0.8rem;
  color: var(--accent);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}
section h2 .icon {{ font-style: normal; }}
.summary p {{ font-size: 1rem; }}
.vocab ul {{ list-style: none; }}
.vocab li {{
  padding: 0.55rem 0;
  border-bottom: 1px dashed var(--border);
  font-size: 0.95rem;
}}
.vocab li:last-child {{ border-bottom: none; }}
.vocab-word {{
  font-weight: 600;
  color: var(--accent);
}}
.example {{
  background: var(--accent-light);
  border-left: 4px solid var(--accent);
}}
.example p, .discussion p {{
  font-size: 1rem;
}}
.source-link {{
  display: block;
  text-align: center;
  margin-top: 0.5rem;
  font-size: 0.85rem;
}}
.source-link a {{
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
}}
.source-link a:hover {{ text-decoration: underline; }}
footer {{
  text-align: center;
  margin-top: 2rem;
  font-size: 0.8rem;
  color: var(--muted);
}}
</style>
</head>
<body>
<div class="container">
  <header>
    <span class="date-badge">{_esc(today)}</span>
    <h1>AI News for English Learners</h1>
    <p>A daily lesson from today's tech headlines</p>
  </header>

  <section class="summary">
    <h2><span class="icon">📰</span> Today's Story</h2>
    <p><strong>{_esc(article['title'])}</strong></p>
    <p style="margin-top:0.6rem">{_esc(lesson['summary'])}</p>
    <p class="source-link"><a href="{_esc(article['link'])}" target="_blank" rel="noopener">Read the original article ↗</a></p>
  </section>

  <section class="vocab">
    <h2><span class="icon">📖</span> Vocabulary</h2>
    <ul>
      {vocab_html}
    </ul>
  </section>

  <section class="example">
    <h2><span class="icon">✏️</span> Example Sentence</h2>
    <p>{_esc(lesson['example_sentence'])}</p>
  </section>

  <section class="discussion">
    <h2><span class="icon">💬</span> Discussion Question</h2>
    <p>{_esc(lesson['discussion_question'])}</p>
  </section>

  <footer>
    Built with Python &amp; GitHub Actions · Updated daily
  </footer>
</div>
</body>
</html>
"""


def build_fallback_html(reason: str = "No article available today.") -> str:
    """Produce a valid placeholder page when the pipeline fails."""
    today = datetime.date.today().strftime("%B %d, %Y")
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI News for English Learners</title>
<style>
body {{
  font-family: system-ui, sans-serif;
  display: flex; align-items: center; justify-content: center;
  min-height: 100vh; margin: 0; background: #f8f5f0; color: #2c2c2c;
  text-align: center; padding: 2rem;
}}
h1 {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
p {{ color: #6b6b6b; }}
</style>
</head>
<body>
<div>
  <h1>AI News for English Learners</h1>
  <p>{_esc(today)}</p>
  <p style="margin-top:1rem">{_esc(reason)}</p>
  <p>Check back tomorrow for a new lesson!</p>
</div>
</body>
</html>
"""


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("▸ Fetching RSS…")
    article = fetch_latest_article()

    if not article:
        print("✖ No article found from any feed.")
        html = build_fallback_html("We couldn't reach any news feeds today.")
        _write(html)
        return

    print(f"▸ Article: {article['title']}")
    print("▸ Calling LLM…")
    lesson = call_llm(article)

    if not lesson:
        print("✖ LLM did not return valid data.")
        html = build_fallback_html("We had trouble processing today's article.")
        _write(html)
        return

    print("▸ Building HTML…")
    html = build_html(article, lesson)
    _write(html)
    print(f"✔ Done — wrote {OUTPUT_FILE}")


def _write(html: str):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
