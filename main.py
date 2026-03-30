"""
AI Agent Daily Digest — v5.0 (LangGraph Multi-Agent Pipeline)

Architecture (LangGraph StateGraph):

  [RSS Fetch] + [Tavily Search]
         |
    [Curator]     Agent 1: Score articles  (flash-lite, 1 call)
         |
    [Selector]    Agent 2: Pick balanced   (flash-lite, 1 call)
         |
    [Deep Reader] Jina Reader: Full text   (HTTP, N calls)
         |
    [Enricher]    Agent 3+4: Summary+Vocab (flash, 2N calls)
         |
    [Build Output] HTML + JSON archive

Total Gemini API calls: ~12  |  Well within free tier (15 RPM)
"""

import os
import json
import glob
import time
import datetime
import re
from typing import TypedDict
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import requests
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

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

MAX_CANDIDATES    = 20
TARGET_ARTICLES   = 5
MIN_ARTICLES      = 3
MAX_ARTICLE_CHARS = 3000
JINA_MAX_CHARS    = 40000
AGENT_SLEEP_SEC   = 5
ENRICH_WORKERS    = 2      # Concurrent enrichment threads (keep <=3 for 15 RPM)

ARCHIVE_DIR  = "archives"
ARCHIVE_DAYS = 30
OUTPUT_FILE  = "index.html"


# ---------------------------------------------------------------------------
# Pipeline State (LangGraph typed state)
# ---------------------------------------------------------------------------
class PipelineState(TypedDict, total=False):
    candidates:       list[dict]
    scored:           list[dict]
    selected:         list[dict]
    final_articles:   list[dict]
    used_vocab_terms: list[str]
    today_str:        str
    status:           str
    output_json:      list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gemini_call(model: str, prompt: str, temperature: float = 0.5) -> str | None:
    if not GEMINI_API_KEY:
        print("X GEMINI_API_KEY is not set.")
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
        print(f"X Gemini call failed ({model}): {exc}")
        return None


def parse_json(text: str | None) -> dict | list | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```[a-z]*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            print("X JSON parse failed")
            return None


def read_article_jina(url: str) -> str | None:
    """Read full article via Jina Reader. Returns None on failure (403, timeout, etc)."""
    try:
        print(f"    [Jina] Reading: {url[:80]}")
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain", "X-No-Cache": "true"},
            timeout=30,
        )
        if resp.status_code == 200:
            text = resp.text[:JINA_MAX_CHARS]
            # Guard against near-empty responses (bot-blocked pages)
            if len(text.strip()) < 200:
                print(f"    [Jina] Response too short ({len(text)} chars), likely blocked: {url[:60]}")
                return None
            return text
        print(f"    [Jina] HTTP {resp.status_code} (blocked/error): {url[:60]}")
    except requests.exceptions.Timeout:
        print(f"    [Jina] Timeout: {url[:60]}")
    except Exception as exc:
        print(f"    [Jina] Error: {exc}")
    return None


def normalize_url(url: str) -> str:
    """Normalize URL for deduplication: strip tracking params, fragments, trailing slashes."""
    try:
        parsed = urlparse(url)
        # Remove common tracking parameters
        tracking_params = {"utm_source", "utm_medium", "utm_campaign", "utm_content",
                           "utm_term", "ref", "source", "fbclid", "gclid"}
        qs = parse_qs(parsed.query, keep_blank_values=False)
        cleaned_qs = {k: v for k, v in qs.items() if k.lower() not in tracking_params}
        clean_query = urlencode(cleaned_qs, doseq=True)
        # Normalize: lowercase host, strip fragment, strip trailing slash
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.params,
            clean_query,
            "",  # drop fragment
        ))
        return normalized
    except Exception:
        return url.lower().strip()


# ---------------------------------------------------------------------------
# Node 1: Fetch Sources (RSS + optional Tavily)
# ---------------------------------------------------------------------------
BROAD_KEYWORDS = [
    "ai", "llm", "model", "agent", "openai", "anthropic", "google",
    "gemini", "gpt", "claude", "machine learning", "deep learning",
    "neural", "automation", "language model", "chatbot", "copilot",
]


def node_fetch_sources(state: PipelineState) -> dict:
    print("\n=== Step 1: Fetching Sources ===")
    candidates: list[dict] = []
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()      # URL-based dedup (Fix #2)

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
                # URL-based dedup: catch same article from different feeds
                norm_url = normalize_url(link)
                if norm_url in seen_urls:
                    continue
                seen_titles.add(norm)
                seen_urls.add(norm_url)
                combined = (title + " " + body).lower()
                if any(kw in combined for kw in BROAD_KEYWORDS):
                    candidates.append({
                        "title": title, "link": link,
                        "body": body[:MAX_ARTICLE_CHARS], "source": "rss",
                    })
                if len(candidates) >= MAX_CANDIDATES:
                    break
        except Exception as exc:
            print(f"  Warning: RSS error ({feed_url}): {exc}")
        if len(candidates) >= MAX_CANDIDATES:
            break

    if TAVILY_API_KEY and len(candidates) < MAX_CANDIDATES:
        print("  [Tavily] Supplementing with web search...")
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": "latest AI LLM agent release news today",
                    "search_depth": "advanced",
                    "time_range": "day",
                    "max_results": 5,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                for r in resp.json().get("results", []):
                    title = r.get("title", "").strip()
                    norm = title.lower().strip()
                    norm_url = normalize_url(r.get("url", ""))
                    if norm and norm not in seen_titles and norm_url not in seen_urls:
                        seen_titles.add(norm)
                        seen_urls.add(norm_url)
                        candidates.append({
                            "title": title, "link": r.get("url", ""),
                            "body": r.get("content", "")[:MAX_ARTICLE_CHARS],
                            "source": "tavily",
                        })
        except Exception as exc:
            print(f"  Warning: Tavily error: {exc}")

    rss_count    = sum(1 for c in candidates if c["source"] == "rss")
    search_count = sum(1 for c in candidates if c["source"] == "tavily")
    print(f"  > Fetched {len(candidates)} candidates ({rss_count} RSS, {search_count} search)")
    return {
        "candidates": candidates,
        "today_str": datetime.date.today().strftime("%Y-%m-%d"),
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Node 2: Agent Curator
# ---------------------------------------------------------------------------
CURATOR_PROMPT = """\
You are an expert AI news curator.
Your audience is Data Scientists and Software Engineers who care about AI agents,
LLMs, and the broader AI ecosystem.

Score EACH of the following articles on a scale of 1-10 for how relevant, timely,
and technically interesting it is for this audience. Be strict.

Return a JSON ARRAY of objects:
[
  {{
    "title": "<exact title as given>",
    "score": <integer 1-10>,
    "reason": "<one sentence: why this score>",
    "category_hint": "release" | "technical" | "use-case" | "industry" | "research"
  }}
]

Rules:
- Return ONLY a valid JSON array.
- Score 8-10: Major model releases, breakthrough research, key agentic framework updates.
- Score 5-7: Interesting but not groundbreaking AI industry news.
- Score 1-4: Tangentially related, business fluff, or non-technical content.
- Penalise opinion pieces with no new information.
- If two articles cover the SAME story (same event, same announcement) from different sources, give the WEAKER one a score of 1 and note "duplicate of [other title]" in the reason. This is critical for deduplication.

Articles:
{articles_block}
"""


def node_agent_curator(state: PipelineState) -> dict:
    candidates = state["candidates"]
    print(f"\n=== Step 2: Agent Curator - Scoring {len(candidates)} articles ===")
    articles_block = ""
    for i, a in enumerate(candidates, 1):
        articles_block += f"\n[{i}] Title: {a['title']}\nText: {a['body'][:500]}\n"
    prompt = CURATOR_PROMPT.format(articles_block=articles_block)
    raw = gemini_call(MODEL_LITE, prompt, temperature=0.2)
    scored = parse_json(raw)
    if not isinstance(scored, list):
        print("  Warning: Curator returned invalid data - using fallback")
        return {
            "scored": [
                {"title": a["title"], "score": 5, "reason": "fallback",
                 "category_hint": "technical", "link": a["link"], "body": a["body"]}
                for a in candidates
            ]
        }
    candidate_map = {a["title"]: a for a in candidates}
    result = []
    for item in scored:
        original = candidate_map.get(item.get("title", ""))
        if original:
            result.append({**item, "link": original["link"], "body": original["body"]})
    print(f"  > Scored {len(result)} articles")
    return {"scored": result}


# ---------------------------------------------------------------------------
# Node 3: Agent Selector
# ---------------------------------------------------------------------------
SELECTOR_PROMPT = """\
You are an AI news editor. Select exactly {n} articles for today's digest.

Selection criteria:
1. Prefer high scores, but DO NOT pick {n} articles of the same category.
2. Aim for category diversity: mix of "release", "technical", "research", "use-case", "industry".
3. If two articles cover the same story, pick the better one only.
4. Today's digest should feel like a well-rounded view of the AI ecosystem.

Return a JSON array of exactly {n} objects:
[
  {{
    "title": "<exact title>",
    "category": "release" | "technical" | "use-case" | "industry" | "research",
    "selection_reason": "<one sentence>"
  }}
]

Rules: Return ONLY a valid JSON array. Exactly {n} items.

Scored articles:
{scored_block}
"""


def node_agent_selector(state: PipelineState) -> dict:
    scored = state["scored"]
    n = TARGET_ARTICLES
    print(f"\n=== Step 3: Agent Selector - Picking {n} from {len(scored)} ===")
    time.sleep(AGENT_SLEEP_SEC)
    scored_block = json.dumps(
        [{"title": s["title"], "score": s.get("score", 5),
          "reason": s.get("reason", ""), "category_hint": s.get("category_hint", "")}
         for s in scored],
        ensure_ascii=False, indent=2,
    )
    prompt = SELECTOR_PROMPT.format(n=n, scored_block=scored_block)
    raw = gemini_call(MODEL_LITE, prompt, temperature=0.3)
    selected_meta = parse_json(raw)
    if not isinstance(selected_meta, list):
        print("  Warning: Selector returned invalid data - using top-scored")
        top = sorted(scored, key=lambda x: x.get("score", 0), reverse=True)[:n]
        return {"selected": [{**a, "category": a.get("category_hint", "technical")} for a in top]}
    scored_map = {s["title"]: s for s in scored}
    result = []
    for item in selected_meta[:n]:
        original = scored_map.get(item.get("title", ""))
        if original:
            result.append({
                "title": original["title"], "link": original["link"],
                "body": original["body"],
                "category": item.get("category", "technical"),
                "selection_reason": item.get("selection_reason", ""),
            })
    print(f"  > Selected {len(result)} articles")
    return {"selected": result}


# ---------------------------------------------------------------------------
# Node 4: Deep Reader (Jina)
# ---------------------------------------------------------------------------
def node_deep_reader(state: PipelineState) -> dict:
    """Read full articles via Jina Reader. Falls back to RSS snippet on failure (Fix #3)."""
    selected = state["selected"]
    print(f"\n=== Step 4: Deep Reader - Fetching full content for {len(selected)} articles ===")
    enriched = []
    for article in selected:
        url = article.get("link", "")
        if url:
            full_text = read_article_jina(url)
            if full_text:
                enriched.append({**article, "body": full_text})
                print(f"    OK: Full content ({len(full_text)} chars)")
                continue
            else:
                print(f"    Fallback: Using RSS snippet for '{article['title'][:50]}'")
        enriched.append(article)
    upgraded = sum(1 for a in enriched if len(a.get("body", "")) > MAX_ARTICLE_CHARS)
    print(f"  > Deep read complete: {upgraded}/{len(enriched)} upgraded, {len(enriched)-upgraded} using RSS fallback")
    return {"selected": enriched}


# ---------------------------------------------------------------------------
# Node 5: Agent Enricher (Summarizer + English Coach)
# ---------------------------------------------------------------------------
SUMMARIZER_PROMPT = """\
You are a senior AI researcher writing for Data Scientists and Software Engineers.

Produce a deep technical summary of the following article.

Return a single JSON object (NOT an array):
{{
  "headline": "A sharp, informative headline (max 12 words, plain English)",
  "one_liner": "One sentence capturing the core news (max 25 words)",
  "detailed_summary": "4-6 sentences. Cover: (1) what happened/released, (2) technical mechanism/architecture, (3) comparison to existing approaches, (4) implications for practitioners. Be specific."
}}

Rules:
- Return ONLY a valid JSON object.
- The detailed_summary must contain at least one concrete technical detail.
- Category context: {category}

Article title: {title}
Article text:
{body}
"""

ENGLISH_COACH_PROMPT = """\
You are an advanced English teacher for technical vocabulary for non-native speakers in DS/SE.

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
  "discussion_question": "One thought-provoking question about the technical or ethical implications."
}}

CRITICAL RULES:
1. Return ONLY a valid JSON object.
2. Exactly 3 vocabulary items.
3. BANNED basic terms: "AI", "LLM", "agent", "open-source", "machine learning", "data", "model", "tool", "system", "feature".
4. CEFR level: B2-C1 or domain-specific jargon.
5. FORBIDDEN terms already used today: {used_terms_list}
6. The discussion question must be specific to this article.

Article title: {title}
Article summary: {summary}
Article text:
{body}
"""


def _enrich_single_article(article: dict, index: int, total: int) -> dict | None:
    """Enrich a single article (summarize + vocab). Runs in a thread."""
    label = f"[{index+1}/{total}]"
    print(f"\n  > {label} {article['title'][:60]}...")

    # --- Agent 3: Summarizer (sends article body to Gemini, NOT the State) ---
    time.sleep(AGENT_SLEEP_SEC)
    summary_prompt = SUMMARIZER_PROMPT.format(
        category=article.get("category", "technical"),
        title=article["title"], body=article["body"],
    )
    summary_raw = gemini_call(MODEL_HEAVY, summary_prompt, temperature=0.4)
    summary = parse_json(summary_raw)
    if not isinstance(summary, dict):
        print(f"    {label} Warning: Summarizer failed - skipping")
        return None
    required_fields = ("headline", "one_liner", "detailed_summary")
    if not all(isinstance(summary.get(k), str) for k in required_fields):
        print(f"    {label} Warning: Summarizer missing fields - skipping")
        return None

    # --- Agent 4: English Coach (uses summary, NOT full body) ---
    time.sleep(AGENT_SLEEP_SEC)
    coach_prompt = ENGLISH_COACH_PROMPT.format(
        used_terms_list="(deduped after collection)",
        title=article["title"],
        summary=summary.get("detailed_summary", ""),
        body=article["body"][:8000],
    )
    coach_raw = gemini_call(MODEL_HEAVY, coach_prompt, temperature=0.6)
    coaching = parse_json(coach_raw)
    if not isinstance(coaching, dict):
        print(f"    {label} Warning: EnglishCoach failed")
        coaching = {"vocabulary": [], "discussion_question": ""}

    print(f"    {label} OK: enriched")
    return {
        "headline":            summary.get("headline", article["title"]),
        "one_liner":           summary.get("one_liner", ""),
        "detailed_summary":    summary.get("detailed_summary", ""),
        "category":            article.get("category", "technical"),
        "vocabulary":          coaching.get("vocabulary", []),
        "discussion_question": coaching.get("discussion_question", ""),
        "source_url":          article.get("link", ""),
        "source_title":        article.get("title", ""),
    }


def node_agent_enricher(state: PipelineState) -> dict:
    """Agent 3+4: Concurrent enrichment with ThreadPoolExecutor (Fix #4).

    Why ThreadPoolExecutor instead of LangGraph Send API:
    - Send API creates separate subgraph states that can't easily share
      used_vocab_terms across parallel branches.
    - With Gemini's 15 RPM limit, we want controlled concurrency (2 workers),
      not unbounded fan-out. ThreadPoolExecutor gives us that control.
    - The vocabulary dedup is applied AFTER collection (fan-in), which is
      simpler and avoids race conditions.
    """
    selected = state["selected"]
    print(f"\n=== Step 5: Agent Enricher - Processing {len(selected)} articles (workers={ENRICH_WORKERS}) ===")

    # --- Concurrent enrichment (Fix #4) ---
    results: list[tuple[int, dict | None]] = []
    with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as executor:
        futures = {
            executor.submit(_enrich_single_article, article, i, len(selected)): i
            for i, article in enumerate(selected)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as exc:
                print(f"    Warning: Article {idx+1} raised exception: {exc}")
                results.append((idx, None))

    # Preserve original order, filter failures
    results.sort(key=lambda x: x[0])
    final_articles = [r for _, r in results if r is not None]

    # --- Vocabulary dedup across all articles (fan-in phase) ---
    used_vocab_terms: set[str] = set(state.get("used_vocab_terms", []))
    for article in final_articles:
        deduped = []
        for v in article.get("vocabulary", []):
            term_key = v.get("term", "").lower().strip()
            if term_key and term_key not in used_vocab_terms:
                used_vocab_terms.add(term_key)
                deduped.append(v)
        article["vocabulary"] = deduped

    # --- Strip raw body from State to free memory (Fix #1) ---
    # The heavy article text was only needed for Gemini prompts (already sent).
    # final_articles only contain lightweight summary JSON, not the 40k-char bodies.

    print(f"\n  > Enrichment complete: {len(final_articles)}/{len(selected)} articles")
    return {
        "final_articles": final_articles,
        "used_vocab_terms": list(used_vocab_terms),
        "output_json": final_articles,
        "selected": [],  # (Fix #1) Clear heavy body text from State
    }


# ---------------------------------------------------------------------------
# Node 6: Build Output
# ---------------------------------------------------------------------------
def node_build_output(state: PipelineState) -> dict:
    final_articles = state["final_articles"]
    today_str = state["today_str"]
    print(f"\n=== Step 6: Build Output - {len(final_articles)} articles ===")
    save_archive(today_str, final_articles)
    cleanup_old_archives()
    archives = load_archives()
    print("  > Building HTML...")
    html = build_html(archives)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  OK: Wrote {OUTPUT_FILE} with {len(archives)} days of archives")
    return {"status": "success"}


def node_write_fallback(state: PipelineState) -> dict:
    print("\n=== Fallback: Writing minimal page ===")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(build_fallback_html())
    return {"status": "fallback"}


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------
def check_candidates(state: PipelineState) -> str:
    count = len(state.get("candidates", []))
    if count < MIN_ARTICLES:
        print(f"  X Only {count} candidates - routing to fallback")
        return "fallback"
    return "continue"

def check_scored(state: PipelineState) -> str:
    count = len(state.get("scored", []))
    if count < MIN_ARTICLES:
        print(f"  X Only {count} scored articles - routing to fallback")
        return "fallback"
    return "continue"

def check_selected(state: PipelineState) -> str:
    count = len(state.get("selected", []))
    if count < MIN_ARTICLES:
        print(f"  X Only {count} selected articles - routing to fallback")
        return "fallback"
    return "continue"

def check_enriched(state: PipelineState) -> str:
    count = len(state.get("final_articles", []))
    if count < MIN_ARTICLES:
        print(f"  X Only {count} enriched articles - routing to fallback")
        return "fallback"
    return "continue"


# ---------------------------------------------------------------------------
# Graph Construction (LangGraph)
# ---------------------------------------------------------------------------
def build_pipeline() -> StateGraph:
    workflow = StateGraph(PipelineState)
    workflow.add_node("fetch_sources",   node_fetch_sources)
    workflow.add_node("agent_curator",   node_agent_curator)
    workflow.add_node("agent_selector",  node_agent_selector)
    workflow.add_node("deep_reader",     node_deep_reader)
    workflow.add_node("agent_enricher",  node_agent_enricher)
    workflow.add_node("build_output",    node_build_output)
    workflow.add_node("write_fallback",  node_write_fallback)
    workflow.set_entry_point("fetch_sources")
    workflow.add_conditional_edges("fetch_sources", check_candidates, {
        "continue": "agent_curator", "fallback": "write_fallback",
    })
    workflow.add_conditional_edges("agent_curator", check_scored, {
        "continue": "agent_selector", "fallback": "write_fallback",
    })
    workflow.add_conditional_edges("agent_selector", check_selected, {
        "continue": "deep_reader", "fallback": "write_fallback",
    })
    workflow.add_edge("deep_reader", "agent_enricher")
    workflow.add_conditional_edges("agent_enricher", check_enriched, {
        "continue": "build_output", "fallback": "write_fallback",
    })
    workflow.add_edge("build_output",  END)
    workflow.add_edge("write_fallback", END)
    return workflow


# ---------------------------------------------------------------------------
# Archive Management
# ---------------------------------------------------------------------------
def save_archive(date_str: str, articles: list[dict]):
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    filepath = os.path.join(ARCHIVE_DIR, f"{date_str}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "articles": articles}, f, ensure_ascii=False, indent=2)
    print(f"  > Saved archive: {filepath}")

def load_archives() -> list[dict]:
    archives = []
    for fp in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.json")), reverse=True):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                archives.append(json.load(f))
        except Exception as exc:
            print(f"  Warning: Error loading {fp}: {exc}")
    return archives

def cleanup_old_archives():
    cutoff = datetime.date.today() - datetime.timedelta(days=ARCHIVE_DAYS)
    for fp in glob.glob(os.path.join(ARCHIVE_DIR, "*.json")):
        basename = os.path.basename(fp).replace(".json", "")
        try:
            if datetime.date.fromisoformat(basename) < cutoff:
                os.remove(fp)
                print(f"  > Removed old archive: {fp}")
        except ValueError:
            continue


# ---------------------------------------------------------------------------
# HTML Generation
# ---------------------------------------------------------------------------
def build_html(archives: list[dict]) -> str:
    today_display = datetime.date.today().strftime("%B %d, %Y")
    archives_json = json.dumps(archives, ensure_ascii=False)
    # The HTML template uses {{ and }} for CSS/JS braces (escaped for f-string)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI Agent Daily Digest</title>
<meta name="description" content="Daily AI agent news digest with English vocabulary lessons."/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root {{
  --bg:#0c0c0e;--surface:#16161a;--surface-hover:#1c1c22;
  --border:#2a2a32;--border-light:#35353f;
  --text:#e8e6e3;--text-secondary:#9a9a9a;--text-muted:#6a6a72;
  --accent:#6ee7b7;--accent-dim:#34d3991a;--accent-mid:#34d39940;
  --release:#818cf8;--release-bg:#818cf815;
  --technical:#f97316;--technical-bg:#f9731615;
  --usecase:#06b6d4;--usecase-bg:#06b6d415;
  --industry:#f472b6;--industry-bg:#f472b615;
  --research:#a78bfa;--research-bg:#a78bfa15;
  --font-display:'Instrument Serif',serif;
  --font-body:'DM Sans',system-ui,sans-serif;
  --font-mono:'JetBrains Mono',monospace;
  --radius:12px;--radius-sm:8px;
}}
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{font-family:var(--font-body);background:var(--bg);color:var(--text);line-height:1.65;-webkit-font-smoothing:antialiased}}
::selection{{background:var(--accent-mid);color:var(--text)}}
.wrapper{{max-width:720px;margin:0 auto;padding:0 1.25rem}}
header{{padding:3rem 0 2rem;border-bottom:1px solid var(--border);margin-bottom:2rem}}
.header-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;flex-wrap:wrap}}
.brand h1{{font-family:var(--font-display);font-size:clamp(1.8rem,5vw,2.6rem);font-weight:400;letter-spacing:-0.02em;line-height:1.1}}
.brand h1 span{{color:var(--accent)}}
.brand p{{font-size:0.875rem;color:var(--text-secondary);margin-top:0.4rem;max-width:360px}}
.date-nav{{display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem}}
.date-nav button{{background:var(--surface);border:1px solid var(--border);color:var(--text-secondary);width:32px;height:32px;border-radius:50%;cursor:pointer;font-size:0.9rem;display:flex;align-items:center;justify-content:center;transition:all .15s}}
.date-nav button:hover{{background:var(--surface-hover);color:var(--text)}}
.date-nav button:disabled{{opacity:.3;cursor:default}}
.date-label{{font-family:var(--font-mono);font-size:.8rem;color:var(--text-secondary);min-width:130px;text-align:center}}
.stats{{display:flex;gap:.75rem;margin-top:1.25rem;flex-wrap:wrap}}
.stat-pill{{display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .7rem;border-radius:50px;font-size:.75rem;font-weight:500;letter-spacing:.02em}}
.cat-release{{background:var(--release-bg);color:var(--release)}}
.cat-technical{{background:var(--technical-bg);color:var(--technical)}}
.cat-use-case{{background:var(--usecase-bg);color:var(--usecase)}}
.cat-industry{{background:var(--industry-bg);color:var(--industry)}}
.cat-research{{background:var(--research-bg);color:var(--research)}}
.cat-dot{{width:6px;height:6px;border-radius:50%;background:currentColor}}
.articles{{display:flex;flex-direction:column;gap:1rem}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;transition:border-color .2s;animation:fadeUp .4s ease both}}
.card:hover{{border-color:var(--border-light)}}
.card:nth-child(1){{animation-delay:.05s}}.card:nth-child(2){{animation-delay:.1s}}
.card:nth-child(3){{animation-delay:.15s}}.card:nth-child(4){{animation-delay:.2s}}
.card:nth-child(5){{animation-delay:.25s}}
.card-header{{padding:1.25rem 1.5rem;cursor:pointer;display:flex;gap:1rem;align-items:flex-start;user-select:none}}
.card-num{{font-family:var(--font-mono);font-size:.7rem;color:var(--text-muted);background:var(--bg);width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px}}
.card-main{{flex:1;min-width:0}}
.card-headline{{font-family:var(--font-display);font-size:1.2rem;font-weight:400;line-height:1.3;margin-bottom:.35rem}}
.card-oneliner{{font-size:.9rem;color:var(--text-secondary);line-height:1.5}}
.card-meta{{display:flex;align-items:center;gap:.6rem;margin-top:.6rem}}
.expand-icon{{color:var(--text-muted);font-size:1rem;transition:transform .25s;flex-shrink:0;margin-top:2px}}
.card.open .expand-icon{{transform:rotate(180deg)}}
.card-detail{{max-height:0;overflow:hidden;transition:max-height .4s ease}}
.card.open .card-detail{{max-height:1400px}}
.detail-inner{{padding:0 1.5rem 1.5rem;padding-left:calc(1.5rem + 28px + 1rem);border-top:1px solid var(--border)}}
.detail-section{{margin-top:1.25rem}}
.detail-section h3{{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:.6rem}}
.detail-summary{{font-size:.95rem;color:var(--text);line-height:1.8}}
.vocab-list{{display:flex;flex-direction:column;gap:.75rem}}
.vocab-item{{background:var(--bg);border-radius:var(--radius-sm);padding:.85rem 1rem}}
.vocab-term{{font-family:var(--font-mono);font-size:.85rem;font-weight:500;color:var(--accent)}}
.vocab-def{{font-size:.85rem;color:var(--text-secondary);margin-top:.2rem}}
.vocab-example{{font-size:.82rem;color:var(--text-muted);font-style:italic;margin-top:.25rem}}
.discussion-box{{background:var(--accent-dim);border:1px solid var(--accent-mid);border-radius:var(--radius-sm);padding:1rem;font-size:.9rem;color:var(--accent);line-height:1.6}}
.source-link{{display:inline-flex;align-items:center;gap:.3rem;font-size:.8rem;color:var(--text-muted);text-decoration:none;margin-top:1rem;transition:color .15s}}
.source-link:hover{{color:var(--accent)}}
.archive-toggle{{background:var(--surface);border:1px solid var(--border);color:var(--text-secondary);font-family:var(--font-body);font-size:.8rem;padding:.45rem 1rem;border-radius:50px;cursor:pointer;transition:all .15s;display:flex;align-items:center;gap:.4rem;margin:2rem auto 0}}
.archive-toggle:hover{{background:var(--surface-hover);color:var(--text)}}
.archive-list{{display:none;margin-top:1rem;padding-bottom:2rem}}
.archive-list.show{{display:block}}
.archive-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:.5rem}}
.archive-item{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:.65rem .8rem;cursor:pointer;transition:all .15s;text-align:center}}
.archive-item:hover{{background:var(--surface-hover);border-color:var(--border-light)}}
.archive-item.active{{border-color:var(--accent);background:var(--accent-dim)}}
.archive-date{{font-family:var(--font-mono);font-size:.75rem;color:var(--text-secondary)}}
.archive-count{{font-size:.7rem;color:var(--text-muted);margin-top:.15rem}}
.empty-state{{text-align:center;padding:4rem 1rem;color:var(--text-muted)}}
.empty-state h2{{font-family:var(--font-display);font-size:1.4rem;color:var(--text-secondary);margin-bottom:.5rem}}
footer{{border-top:1px solid var(--border);margin-top:3rem;padding:1.5rem 0 2rem;text-align:center;font-size:.75rem;color:var(--text-muted)}}
.pipeline-badge{{font-family:var(--font-mono);font-size:.65rem;color:var(--text-muted);margin-top:.2rem}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
@media(max-width:540px){{
  header{{padding:2rem 0 1.5rem}}
  .detail-inner{{padding-left:1.25rem;padding-right:1.25rem}}
  .card-header{{padding:1rem 1.15rem}}
  .card-num{{display:none}}
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
  <footer>
    AI Agent Daily Digest &mdash; LangGraph multi-agent pipeline &bull; Gemini &bull; GitHub Actions
    <div class="pipeline-badge">v5.0 &middot; LangGraph StateGraph &middot; Jina Reader &middot; 4-agent pipeline</div>
  </footer>
</div>
<script>
const ARCHIVES={archives_json};
let currentIndex=0;
const categoryMeta={{
  'release':{{label:'Release',cls:'cat-release'}},
  'technical':{{label:'Technical',cls:'cat-technical'}},
  'use-case':{{label:'Use case',cls:'cat-use-case'}},
  'industry':{{label:'Industry',cls:'cat-industry'}},
  'research':{{label:'Research',cls:'cat-research'}},
}};
function esc(s){{const d=document.createElement('div');d.textContent=s||'';return d.innerHTML}}
function renderDay(index){{
  currentIndex=index;
  const data=ARCHIVES[index];
  const container=document.getElementById('articles-container');
  const statsBar=document.getElementById('stats-bar');
  const dateLabel=document.getElementById('date-label');
  document.getElementById('prev-btn').disabled=index>=ARCHIVES.length-1;
  document.getElementById('next-btn').disabled=index<=0;
  if(!data||!data.articles||!data.articles.length){{
    container.innerHTML='<div class="empty-state"><h2>No articles today</h2><p>Check back tomorrow.</p></div>';
    statsBar.innerHTML='';dateLabel.textContent=data?data.date:'No data';return;
  }}
  try{{const d=new Date(data.date+'T00:00:00');dateLabel.textContent=d.toLocaleDateString('en-US',{{year:'numeric',month:'long',day:'numeric'}})}}catch(e){{dateLabel.textContent=data.date}}
  const cats={{}};
  data.articles.forEach(a=>{{const c=a.category||'technical';cats[c]=(cats[c]||0)+1}});
  statsBar.innerHTML=
    `<span class="stat-pill" style="background:var(--accent-dim);color:var(--accent);">${{data.articles.length}} article${{data.articles.length>1?'s':''}}</span>`+
    Object.entries(cats).map(([cat,count])=>{{
      const m=categoryMeta[cat]||categoryMeta['technical'];
      return `<span class="stat-pill ${{m.cls}}"><span class="cat-dot"></span>${{m.label}} ${{count}}</span>`;
    }}).join('');
  container.innerHTML=data.articles.map((article,i)=>{{
    const cat=article.category||'technical';
    const m=categoryMeta[cat]||categoryMeta['technical'];
    const vocabHtml=(article.vocabulary||[]).map(v=>
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
        <div class="detail-section"><h3>Summary</h3><div class="detail-summary">${{esc(article.detailed_summary)}}</div></div>
        <div class="detail-section"><h3>Vocabulary</h3><div class="vocab-list">${{vocabHtml}}</div></div>
        <div class="detail-section"><h3>Discussion</h3><div class="discussion-box">${{esc(article.discussion_question)}}</div></div>
        ${{article.source_url?`<a class="source-link" href="${{esc(article.source_url)}}" target="_blank" rel="noopener">Read original article &#8599;</a>`:''}}
      </div></div>
    </div>`;
  }}).join('');
  renderArchiveGrid();
}}
function toggleCard(i){{const c=document.getElementById('card-'+i);if(c)c.classList.toggle('open')}}
function navDate(dir){{const n=currentIndex-dir;if(n>=0&&n<ARCHIVES.length)renderDay(n)}}
function toggleArchive(){{document.getElementById('archive-list').classList.toggle('show')}}
function renderArchiveGrid(){{
  const grid=document.getElementById('archive-grid');
  grid.innerHTML=ARCHIVES.map((a,i)=>{{
    const count=a.articles?a.articles.length:0;
    return `<div class="archive-item ${{i===currentIndex?'active':''}}" onclick="renderDay(${{i}});window.scrollTo(0,0);">
      <div class="archive-date">${{a.date}}</div>
      <div class="archive-count">${{count}} article${{count!==1?'s':''}}</div>
    </div>`;
  }}).join('');
}}
if(ARCHIVES.length>0){{renderDay(0)}}
else{{document.getElementById('articles-container').innerHTML='<div class="empty-state"><h2>Welcome</h2><p>The first digest will appear after the next scheduled run.</p></div>'}}
renderArchiveGrid();
</script>
</body>
</html>"""


def build_fallback_html() -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        '<title>AI Agent Daily Digest</title>'
        '<style>body{font-family:system-ui,sans-serif;display:flex;align-items:center;'
        'justify-content:center;min-height:100vh;margin:0;background:#0c0c0e;color:#e8e6e3;'
        'text-align:center;padding:2rem}h1{font-size:1.4rem;margin-bottom:.5rem}p{color:#9a9a9a}'
        '</style></head><body><div><h1>AI Agent Daily Digest</h1>'
        '<p style="margin-top:1rem">We had trouble generating today\'s digest.</p>'
        '<p>Check back tomorrow!</p></div></body></html>'
    )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  AI Agent Daily Digest - v5.0 (LangGraph Pipeline)")
    print("=" * 60)

    workflow = build_pipeline()
    app = workflow.compile()

    initial_state: PipelineState = {
        "candidates": [], "scored": [], "selected": [],
        "final_articles": [], "used_vocab_terms": [],
        "today_str": "", "status": "running", "output_json": [],
    }

    final_state = None
    for state_snapshot in app.stream(initial_state, stream_mode="values"):
        final_state = state_snapshot

    print("\n" + "=" * 60)
    if final_state and final_state.get("status") == "success":
        articles = final_state.get("output_json", [])
        print(f"  OK: Pipeline SUCCESS - {len(articles)} articles published")
        print(f"  OK: HTML: {OUTPUT_FILE}")
        print(f"  OK: Archive: {ARCHIVE_DIR}/")
        # output_json is ready for downstream: Slack, DB, API, etc.
    else:
        print("  Warning: Pipeline completed with fallback")
    print("=" * 60)


if __name__ == "__main__":
    main()
