"""
AI Agent Daily Digest — v6.0 (ReAct Curation Pipeline)

Architecture (LangGraph StateGraph with ReAct loop):

  [Fetch Sources]  RSS + Tavily + cross-day dedup
         |
  [Triage]         Agent 1: Quick-scan all candidates       (flash-lite, 1 call)
         |
  ┌─[Investigate]  Jina Reader: Full text for shortlist     (HTTP, ~10 calls)
  │      |
  │ [Deep Score]   Agent 2: 2-axis scoring per article      (flash + thinking, N calls)
  │      |
  │ [Reflect]      Agent 3: Self-critique, decide next step (flash, 1 call)
  │      |
  └──< if needs more candidates, loop back to Investigate >
         |
  [Select]         Agent 4: Final pick with constraints     (flash-lite, 1 call)
         |
  [Enricher]       Agent 5+6: Summary + Vocab per article   (flash, 2N calls)
         |
  [Build Output]   HTML + JSON archive

Estimated cost: ~$0.05-0.10/day ≈ ~300-500 yen/month (Gemini 2.5 Flash)
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
    # --- General AI News ---
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.technologyreview.com/feed/",
    # --- Vendor / Research blogs ---
    "https://blog.google/technology/ai/rss/",
    "https://openai.com/blog/rss.xml",
    # --- Practical AI / Engineering / Use-case ---
    "https://ai.gopubby.com/feed",                         # AI in Plain English
    "https://towardsdatascience.com/feed",                  # Towards Data Science
    "https://neptune.ai/blog/feed",                         # Neptune.ai — MLOps
    "https://www.deeplearning.ai/the-batch/feed/",          # Andrew Ng's The Batch
    "https://huyenchip.com/feed.xml",                       # Chip Huyen — ML systems
    "https://simonwillison.net/atom/everything/",           # Simon Willison — practical LLM
    "https://lilianweng.github.io/index.xml",               # Lil'Log — deep practical AI
    "https://www.latent.space/feed",                        # Latent Space podcast/blog
    "https://newsletter.pragmaticengineer.com/feed",        # Pragmatic Engineer × AI
]

MAX_CANDIDATES      = 30
TARGET_ARTICLES     = 5
MIN_ARTICLES        = 3
MAX_ARTICLE_CHARS   = 3000
JINA_MAX_CHARS      = 40000
AGENT_SLEEP_SEC     = 3
ENRICH_WORKERS      = 2
INVESTIGATE_WORKERS = 3

ARCHIVE_DIR         = "archives"
ARCHIVE_DAYS        = 30
OUTPUT_FILE         = "index.html"

# ReAct curation config
DEDUP_LOOKBACK_DAYS   = 3
REACT_MAX_ITERATIONS  = 2
SHORTLIST_SIZE        = 10
THINKING_BUDGET_DEEP  = 8192   # tokens for deep scoring (higher = more thoughtful)

# 2-axis scoring weights (business_value-heavy per user preference)
WEIGHT_WORLD    = 0.3
WEIGHT_BUSINESS = 0.7

BROAD_KEYWORDS = [
    # Core AI terms
    "ai", "llm", "model", "agent", "openai", "anthropic", "google",
    "gemini", "gpt", "claude", "machine learning", "deep learning",
    "neural", "automation", "language model", "chatbot", "copilot",
    # Practical use-case / infrastructure / evaluation
    "use case", "production", "infrastructure", "evaluation", "roi",
    "deployment", "efficiency", "data pipeline", "case study", "benchmark",
    # Practical engineering keywords
    "mlops", "fine-tune", "fine-tuning", "rag", "retrieval", "vector",
    "embedding", "prompt engineering", "guardrail", "observability",
    "cost reduction", "latency", "real-world", "lessons learned",
    "postmortem", "architecture", "workflow", "pipeline",
]


# ---------------------------------------------------------------------------
# Pipeline State (LangGraph typed state)
# ---------------------------------------------------------------------------
class PipelineState(TypedDict, total=False):
    candidates:        list[dict]
    today_str:         str
    status:            str
    # Cross-day dedup
    recent_urls:       list[str]
    recent_headlines:  list[str]
    # ReAct curation loop
    triage_scored:     list[dict]     # ALL candidates with triage scores (for expand)
    shortlist:         list[dict]     # after triage
    investigated:      list[dict]     # after Jina read (accumulates)
    deep_scored:       list[dict]     # after 2-axis scoring (accumulates)
    react_iteration:   int            # 0-based, incremented by reflect
    react_action:      str            # "investigate_more" | "finalize"
    # Selection & enrichment
    selected:          list[dict]
    final_articles:    list[dict]
    used_vocab_terms:  list[str]
    output_json:       list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gemini_call(
    model: str,
    prompt: str,
    temperature: float = 0.5,
    thinking_budget: int | None = None,
) -> str | None:
    """Call Gemini API. Optional thinking_budget for deep reasoning."""
    if not GEMINI_API_KEY:
        print("✗ GEMINI_API_KEY is not set.")
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )
    gen_config: dict = {
        "temperature": temperature,
        "responseMimeType": "application/json",
    }
    if thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": gen_config,
            },
            timeout=180,
        )
        resp.raise_for_status()
        result = resp.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        print(f"✗ Gemini call failed ({model}): {exc}")
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
            print("✗ JSON parse failed")
            return None


def read_article_jina(url: str) -> str | None:
    """Read full article via Jina Reader."""
    try:
        print(f"    [Jina] Reading: {url[:80]}")
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain", "X-No-Cache": "true"},
            timeout=30,
        )
        if resp.status_code == 200:
            text = resp.text[:JINA_MAX_CHARS]
            if len(text.strip()) < 200:
                print(f"    [Jina] Response too short ({len(text)} chars): {url[:60]}")
                return None
            return text
        print(f"    [Jina] HTTP {resp.status_code}: {url[:60]}")
    except requests.exceptions.Timeout:
        print(f"    [Jina] Timeout: {url[:60]}")
    except Exception as exc:
        print(f"    [Jina] Error: {exc}")
    return None


def normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    try:
        parsed = urlparse(url)
        tracking_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_content",
            "utm_term", "ref", "source", "fbclid", "gclid",
        }
        qs = parse_qs(parsed.query, keep_blank_values=False)
        cleaned_qs = {k: v for k, v in qs.items() if k.lower() not in tracking_params}
        clean_query = urlencode(cleaned_qs, doseq=True)
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.params,
            clean_query,
            "",
        ))
        return normalized
    except Exception:
        return url.lower().strip()


def load_recent_article_fingerprints(
    lookback_days: int = DEDUP_LOOKBACK_DAYS,
) -> tuple[set[str], list[str]]:
    """Load URLs and headlines from recent archives for cross-day dedup."""
    recent_urls: set[str] = set()
    recent_headlines: list[str] = []
    today = datetime.date.today()

    for days_ago in range(1, lookback_days + 1):
        date_str = (today - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")
        filepath = os.path.join(ARCHIVE_DIR, f"{date_str}.json")
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                archive = json.load(f)
            for article in archive.get("articles", []):
                url = article.get("source_url", "")
                if url:
                    recent_urls.add(normalize_url(url))
                for key in ("headline", "source_title"):
                    val = article.get(key, "")
                    if val:
                        recent_headlines.append(val)
        except Exception as exc:
            print(f"  Warning: Error loading {filepath}: {exc}")

    return recent_urls, recent_headlines


# ---------------------------------------------------------------------------
# Node 1: Fetch Sources (RSS + optional Tavily + cross-day dedup)
# ---------------------------------------------------------------------------
def node_fetch_sources(state: PipelineState) -> dict:
    print("\n=== Step 1: Fetching Sources ===")

    # Cross-day dedup: load recent article fingerprints
    recent_urls, recent_headlines = load_recent_article_fingerprints()
    recent_titles_lower = {h.lower().strip() for h in recent_headlines}
    print(f"  > Cross-day dedup: {len(recent_urls)} URLs, "
          f"{len(recent_titles_lower)} headlines from last {DEDUP_LOOKBACK_DAYS} days")

    candidates: list[dict] = []
    seen_titles: set[str] = set()
    seen_urls: set[str] = set(recent_urls)  # seed with past URLs

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
                # Cross-day title dedup
                if norm in recent_titles_lower:
                    continue
                # URL-based dedup
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

    # Tavily supplementation
    if TAVILY_API_KEY and len(candidates) < MAX_CANDIDATES:
        print("  [Tavily] Supplementing with web search...")
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": (
                        "latest AI LLM agent (production deployment OR "
                        "architecture OR case study OR evaluation OR MLOps) today"
                    ),
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
                    if (norm and norm not in seen_titles
                            and norm not in recent_titles_lower
                            and norm_url not in seen_urls):
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
    print(f"  > Fetched {len(candidates)} candidates "
          f"({rss_count} RSS, {search_count} search)")
    return {
        "candidates": candidates,
        "today_str": datetime.date.today().strftime("%Y-%m-%d"),
        "status": "running",
        "recent_urls": list(recent_urls),
        "recent_headlines": recent_headlines,
        "react_iteration": 0,
        "react_action": "",
        "investigated": [],
        "deep_scored": [],
    }


# ---------------------------------------------------------------------------
# Node 2: Triage — quick-scan, build shortlist (flash-lite, 1 call)
# ---------------------------------------------------------------------------
TRIAGE_PROMPT = """\
You are an AI news curator. Quick-scan these article candidates and score each 1-10.

Audience: Data Scientists and Software Engineers who want PRACTICAL, ACTIONABLE
AI/ML engineering content — not a product announcement feed.

Return a JSON ARRAY of objects:
[
  {{
    "title": "<exact title as given>",
    "score": <integer 1-10>,
    "reason": "<one sentence>"
  }}
]

Scoring guide:
- 8-10: Deep technical/practical content (production deployments, architecture,
  evaluation, cost analysis, MLOps, real-world case studies), OR truly major
  industry events (new flagship models, paradigm shifts).
- 5-7: Interesting AI news with some technical substance.
- 3-4: Minor product updates, version bumps, incremental features, funding rounds
  with no technical angle.
- 1-2: PR fluff, opinion with no data, tangentially related.

RELEASE PENALTY: Minor releases (feature updates, v2.x patches, API additions,
small model variants) should score 3-5 max. Only truly major releases score 8+.

DEDUP: If two articles clearly cover the same story, score the weaker one 1.

Articles:
{articles_block}
"""


def node_triage(state: PipelineState) -> dict:
    candidates = state["candidates"]
    print(f"\n=== Step 2: Triage — scoring {len(candidates)} candidates ===")

    articles_block = ""
    for i, a in enumerate(candidates, 1):
        articles_block += f"\n[{i}] Title: {a['title']}\nSnippet: {a['body'][:400]}\n"

    prompt = TRIAGE_PROMPT.format(articles_block=articles_block)
    raw = gemini_call(MODEL_LITE, prompt, temperature=0.2)
    scored = parse_json(raw)

    if not isinstance(scored, list):
        print("  Warning: Triage failed — using all candidates as shortlist")
        return {"shortlist": candidates[:SHORTLIST_SIZE]}

    # Map scores back to candidates
    candidate_map = {a["title"]: a for a in candidates}
    scored_candidates = []
    for item in scored:
        original = candidate_map.get(item.get("title", ""))
        if original:
            scored_candidates.append({
                **original,
                "triage_score": item.get("score", 5),
                "triage_reason": item.get("reason", ""),
            })

    # Sort by triage score, take top SHORTLIST_SIZE
    scored_candidates.sort(key=lambda x: x.get("triage_score", 0), reverse=True)
    shortlist = scored_candidates[:SHORTLIST_SIZE]

    print(f"  > Shortlisted {len(shortlist)} articles "
          f"(scores: {[a['triage_score'] for a in shortlist]})")
    # Keep ALL scored candidates for potential ReAct expansion (sorted by score)
    return {"shortlist": shortlist, "triage_scored": scored_candidates}


# ---------------------------------------------------------------------------
# Node 3: Investigate — Jina Reader full text (parallel HTTP)
# ---------------------------------------------------------------------------
def node_investigate(state: PipelineState) -> dict:
    shortlist = state["shortlist"]
    # Build dedup set from already-investigated articles, keyed by normalized URL
    # Filter out empty URLs to prevent false matches
    already_investigated = {
        normalize_url(a["link"])
        for a in state.get("investigated", [])
        if a.get("link", "").strip()
    }
    to_investigate = [
        a for a in shortlist
        if a.get("link", "").strip()
        and normalize_url(a["link"]) not in already_investigated
    ]

    print(f"\n=== Step 3: Investigate — reading {len(to_investigate)} articles "
          f"(already done: {len(already_investigated)}) ===")

    def _read_one(article: dict) -> dict:
        url = article.get("link", "")
        full_text = read_article_jina(url) if url else None
        if full_text:
            return {**article, "full_text": full_text}
        else:
            print(f"    Fallback: RSS snippet for '{article['title'][:50]}'")
            return {**article, "full_text": article.get("body", "")}

    newly_investigated: list[dict] = []
    with ThreadPoolExecutor(max_workers=INVESTIGATE_WORKERS) as executor:
        futures = {executor.submit(_read_one, a): a for a in to_investigate}
        for future in as_completed(futures):
            try:
                newly_investigated.append(future.result())
            except Exception as exc:
                article = futures[future]
                print(f"    Warning: Failed to read '{article['title'][:40]}': {exc}")
                newly_investigated.append({**article, "full_text": article.get("body", "")})

    previous = state.get("investigated", [])
    all_investigated = previous + newly_investigated
    upgraded = sum(1 for a in newly_investigated if len(a.get("full_text", "")) > MAX_ARTICLE_CHARS)
    print(f"  > Investigated: {upgraded}/{len(newly_investigated)} upgraded with full text, "
          f"total pool: {len(all_investigated)}")
    return {"investigated": all_investigated}


# ---------------------------------------------------------------------------
# Node 4: Deep Score — 2-axis scoring with full text (flash + thinking)
# ---------------------------------------------------------------------------
DEEP_SCORE_PROMPT = """\
You are a senior AI/ML technical evaluator. Score this article on TWO independent axes.

AXIS 1 — World Importance (世間的な重要度・温度感):
How significant is this for the global AI/tech community?
- 9-10: Industry-defining (new flagship model, major regulation, paradigm shift)
- 7-8: Significant (major funding, notable research paper, important policy)
- 5-6: Noteworthy but not major (company updates, smaller research)
- 3-4: Minor (small updates, opinions, general commentary)
- 1-2: Trivial or tangentially related

AXIS 2 — Business Value (ビジネス活用度):
How actionable and useful for DS/SE practitioners in their daily work?
- 9-10: Directly implementable — specific architecture, code patterns, metrics, benchmarks
- 7-8: Clear lessons, frameworks, or approaches applicable to real projects
- 5-6: Useful context but no direct action items
- 3-4: Awareness-only, no practical application
- 1-2: No business relevance

RELEASE ARTICLES:
- If this is a minor release (feature update, version bump, API addition, small
  model variant, pricing change): world_importance max 5, business_value max 4.
- Only truly MAJOR releases (new flagship model family, paradigm-shifting
  framework) deserve high scores.

DUPLICATE CHECK against recent headlines:
{recent_headlines_block}

Return a single JSON object:
{{
  "world_importance": <int 1-10>,
  "world_reason": "<one sentence>",
  "business_value": <int 1-10>,
  "business_reason": "<one sentence>",
  "category": "release" | "technical" | "use-case" | "industry" | "research",
  "release_magnitude": "major" | "minor" | "none",
  "is_duplicate_of_recent": <bool>,
  "duplicate_note": "<which recent headline, or empty>",
  "key_insight": "<one sentence: what a practitioner would learn from this>"
}}

Article title: {title}
Article text (first 6000 chars):
{body}
"""


def _deep_score_one(
    article: dict, index: int, total: int, recent_headlines_block: str,
) -> dict | None:
    """Score a single article on 2 axes. Runs in a thread."""
    label = f"[{index+1}/{total}]"
    print(f"\n  > {label} Scoring: {article['title'][:60]}...")
    time.sleep(AGENT_SLEEP_SEC)

    body_text = article.get("full_text", article.get("body", ""))
    prompt = DEEP_SCORE_PROMPT.format(
        title=article["title"],
        body=body_text[:6000],
        recent_headlines_block=recent_headlines_block,
    )
    raw = gemini_call(
        MODEL_HEAVY, prompt,
        temperature=0.2,
        thinking_budget=THINKING_BUDGET_DEEP,
    )
    result = parse_json(raw)
    if not isinstance(result, dict):
        print(f"    {label} Warning: Deep score failed — using triage score")
        triage = article.get("triage_score", 5)
        return {
            **article,
            "world_importance": triage,
            "world_reason": "(fallback: deep score failed)",
            "business_value": triage,
            "business_reason": "(fallback: deep score failed)",
            "combined_score": triage,
            "category": "technical",
            "release_magnitude": "none",
            "is_duplicate_of_recent": False,
            "duplicate_note": "",
            "key_insight": "",
        }

    world = result.get("world_importance", 5)
    biz   = result.get("business_value", 5)

    # Apply release penalty
    if result.get("release_magnitude") == "minor":
        world = min(world, 5)
        biz   = min(biz, 4)

    # Duplicate → score 0
    if result.get("is_duplicate_of_recent"):
        print(f"    {label} DUPLICATE of recent: {result.get('duplicate_note', '')}")
        world = 0
        biz   = 0

    combined = round(WEIGHT_WORLD * world + WEIGHT_BUSINESS * biz, 2)
    print(f"    {label} ✓ world={world} biz={biz} combined={combined} "
          f"cat={result.get('category', '?')}")

    return {
        **article,
        "world_importance": world,
        "business_value": biz,
        "combined_score": combined,
        "category": result.get("category", "technical"),
        "release_magnitude": result.get("release_magnitude", "none"),
        "is_duplicate_of_recent": result.get("is_duplicate_of_recent", False),
        "key_insight": result.get("key_insight", ""),
        "world_reason": result.get("world_reason", ""),
        "business_reason": result.get("business_reason", ""),
    }


def node_deep_score(state: PipelineState) -> dict:
    investigated = state["investigated"]
    already_scored_titles = {a["title"] for a in state.get("deep_scored", [])}
    to_score = [a for a in investigated if a["title"] not in already_scored_titles]

    print(f"\n=== Step 4: Deep Score — scoring {len(to_score)} articles "
          f"(2-axis, thinking_budget={THINKING_BUDGET_DEEP}) ===")

    recent_headlines = state.get("recent_headlines", [])
    if recent_headlines:
        recent_headlines_block = "\n".join(f"  - {h}" for h in recent_headlines[:20])
    else:
        recent_headlines_block = "  (none — first day)"

    # Sequential scoring to respect rate limits, with controlled concurrency
    results: list[tuple[int, dict | None]] = []
    with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as executor:
        futures = {
            executor.submit(
                _deep_score_one, article, i, len(to_score), recent_headlines_block,
            ): i
            for i, article in enumerate(to_score)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append((idx, future.result()))
            except Exception as exc:
                print(f"    Warning: Scoring article {idx+1} failed: {exc}")
                results.append((idx, None))

    results.sort(key=lambda x: x[0])
    newly_scored = [r for _, r in results if r is not None]

    # Filter out duplicates of recent articles
    newly_scored = [a for a in newly_scored if not a.get("is_duplicate_of_recent")]

    previous = state.get("deep_scored", [])
    all_scored = previous + newly_scored
    all_scored.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

    print(f"\n  > Deep scored: {len(newly_scored)} new, {len(all_scored)} total")
    for a in all_scored[:8]:
        print(f"    {a['combined_score']:.1f} [{a.get('category','?')[:3]}] "
              f"{a['title'][:55]}")
    return {"deep_scored": all_scored}


# ---------------------------------------------------------------------------
# Node 5: Reflect — self-critique, decide loop or finalize (ReAct core)
# ---------------------------------------------------------------------------
REFLECT_PROMPT = """\
You are the editorial director reviewing curation results for an AI news digest.

Current scored articles (sorted by combined score):
{scored_summary}

REQUIREMENTS for a good digest:
- {n} articles total
- MAX 1 release article (only if it's a truly major release, score >= 7)
- MIN 2 practical/engineering articles (category = "use-case" or "technical")
- Good category diversity (at least 3 different categories)
- All articles should have combined_score >= 5.0
- No duplicates with recent days

Candidates still available but NOT yet investigated:
  {remaining_count} candidates remain in the original pool.

Answer these questions, then decide:
1. Do we have at least {n} articles with combined_score >= 5.0?
2. Do we have >= 2 practical/engineering articles with business_value >= 7?
3. Is category diversity acceptable (>= 3 categories)?
4. Any quality concerns?

Return JSON:
{{
  "assessment": "<2-3 sentences summarizing quality>",
  "quality_sufficient": <bool>,
  "practical_count": <int — articles with category use-case/technical AND biz_value>=7>,
  "category_count": <int — distinct categories in top {n}>,
  "action": "finalize" | "investigate_more",
  "expand_by": <int, how many more candidates to add if action=investigate_more, 0 otherwise>
}}
"""


def node_reflect(state: PipelineState) -> dict:
    deep_scored = state.get("deep_scored", [])
    iteration = state.get("react_iteration", 0) + 1
    candidates = state["candidates"]
    shortlist = state["shortlist"]
    shortlisted_titles = {a["title"] for a in shortlist}

    remaining_count = sum(1 for c in candidates if c["title"] not in shortlisted_titles)

    print(f"\n=== Step 5: Reflect — iteration {iteration}/{REACT_MAX_ITERATIONS} ===")

    # Build scored summary for prompt
    scored_summary = json.dumps(
        [{
            "title": a["title"][:60],
            "combined_score": a.get("combined_score", 0),
            "world_importance": a.get("world_importance", 0),
            "business_value": a.get("business_value", 0),
            "category": a.get("category", "?"),
            "release_magnitude": a.get("release_magnitude", "none"),
        } for a in deep_scored[:12]],
        ensure_ascii=False, indent=2,
    )

    time.sleep(AGENT_SLEEP_SEC)
    prompt = REFLECT_PROMPT.format(
        scored_summary=scored_summary,
        n=TARGET_ARTICLES,
        remaining_count=remaining_count,
    )
    raw = gemini_call(MODEL_HEAVY, prompt, temperature=0.3, thinking_budget=4096)
    reflection = parse_json(raw)

    if not isinstance(reflection, dict):
        print("  Warning: Reflect failed — proceeding to select")
        return {"react_iteration": iteration, "react_action": "finalize"}

    action = reflection.get("action", "finalize")
    expand_by = reflection.get("expand_by", 0)
    print(f"  > Assessment: {reflection.get('assessment', 'N/A')}")
    print(f"  > Quality sufficient: {reflection.get('quality_sufficient')}")
    print(f"  > Practical articles: {reflection.get('practical_count', '?')}")
    print(f"  > Category diversity: {reflection.get('category_count', '?')} categories")
    print(f"  > Action: {action} (expand_by={expand_by})")

    # If agent wants more and we haven't hit max iterations
    if action == "investigate_more" and iteration < REACT_MAX_ITERATIONS and remaining_count > 0:
        # Expand shortlist with next-best candidates, sorted by triage score
        expand_by = min(expand_by or 5, remaining_count, 5)
        # Use triage_scored (pre-sorted by score) instead of raw candidates
        triage_scored = state.get("triage_scored", [])
        remaining = [
            c for c in triage_scored
            if c["title"] not in shortlisted_titles
        ]
        new_additions = remaining[:expand_by]
        if not new_additions:
            # triage_scored empty (e.g. triage fallback) — fall back to candidates
            remaining_raw = [c for c in candidates if c["title"] not in shortlisted_titles]
            new_additions = remaining_raw[:expand_by]
        expanded_shortlist = shortlist + new_additions
        print(f"  > Expanding shortlist by {len(new_additions)} "
              f"(triage scores: {[a.get('triage_score', '?') for a in new_additions]}, "
              f"total: {len(expanded_shortlist)})")
        return {
            "shortlist": expanded_shortlist,
            "react_iteration": iteration,
            "react_action": "investigate_more",
        }

    return {"react_iteration": iteration, "react_action": "finalize"}


def should_continue_react(state: PipelineState) -> str:
    """Conditional edge: loop back to investigate or proceed to select."""
    if state.get("react_iteration", 0) >= REACT_MAX_ITERATIONS:
        return "select"
    if state.get("react_action") == "investigate_more":
        return "investigate"
    return "select"


# ---------------------------------------------------------------------------
# Node 6: Select — final selection with constraints (flash-lite, 1 call)
# ---------------------------------------------------------------------------
SELECT_PROMPT = """\
You are an AI news editor. Select exactly {n} articles for today's digest.

HARD CONSTRAINTS:
1. "release" category: MAX 1 article. Only if combined_score >= 7.0 AND
   release_magnitude = "major". If no release qualifies, pick ZERO releases.
2. "use-case" + "technical" combined: AT LEAST 2 articles.
3. Category diversity: at least 3 different categories among the {n} picks.
4. Prefer higher combined_score, but respect the constraints above.
5. No two articles about substantially the same topic.

Return a JSON array of exactly {n} objects:
[
  {{
    "title": "<exact title>",
    "category": "release" | "technical" | "use-case" | "industry" | "research",
    "selection_reason": "<one sentence>"
  }}
]

Scored articles (sorted by combined_score):
{scored_block}
"""


def node_select(state: PipelineState) -> dict:
    deep_scored = state.get("deep_scored", [])
    n = TARGET_ARTICLES

    print(f"\n=== Step 6: Select — picking {n} from {len(deep_scored)} scored ===")
    time.sleep(AGENT_SLEEP_SEC)

    scored_block = json.dumps(
        [{
            "title": a["title"],
            "combined_score": a.get("combined_score", 0),
            "world_importance": a.get("world_importance", 0),
            "business_value": a.get("business_value", 0),
            "category": a.get("category", "technical"),
            "release_magnitude": a.get("release_magnitude", "none"),
            "key_insight": a.get("key_insight", ""),
        } for a in deep_scored[:15]],
        ensure_ascii=False, indent=2,
    )

    prompt = SELECT_PROMPT.format(n=n, scored_block=scored_block)
    raw = gemini_call(MODEL_LITE, prompt, temperature=0.3)
    selected_meta = parse_json(raw)

    if not isinstance(selected_meta, list):
        print("  Warning: Selector failed — using top combined_score")
        top = deep_scored[:n]
        return {"selected": [{
            **a, "category": a.get("category", "technical"),
        } for a in top]}

    scored_map = {a["title"]: a for a in deep_scored}
    result = []
    for item in selected_meta[:n]:
        original = scored_map.get(item.get("title", ""))
        if original:
            result.append({
                "title": original["title"],
                "link": original.get("link", ""),
                "body": original.get("full_text", original.get("body", "")),
                "category": item.get("category", "technical"),
                "selection_reason": item.get("selection_reason", ""),
            })

    print(f"  > Selected {len(result)} articles:")
    for a in result:
        print(f"    [{a['category'][:3]}] {a['title'][:60]}")

    # --- State GC: strip heavy full_text from investigated/deep_scored ---
    # These are no longer needed; selected articles already carry their body.
    selected_titles = {a["title"] for a in result}
    gc_investigated = [
        {k: v for k, v in a.items() if k != "full_text"}
        for a in state.get("investigated", [])
        if a["title"] not in selected_titles
    ]
    gc_deep_scored = [
        {k: v for k, v in a.items() if k != "full_text"}
        for a in state.get("deep_scored", [])
    ]
    print(f"  > GC: stripped full_text from {len(gc_investigated)} investigated + "
          f"{len(gc_deep_scored)} scored articles")
    return {"selected": result, "investigated": gc_investigated, "deep_scored": gc_deep_scored}


# ---------------------------------------------------------------------------
# Node 7: Agent Enricher (Summarizer + English Coach)
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

    time.sleep(AGENT_SLEEP_SEC)
    summary_prompt = SUMMARIZER_PROMPT.format(
        category=article.get("category", "technical"),
        title=article["title"], body=article["body"],
    )
    summary_raw = gemini_call(MODEL_HEAVY, summary_prompt, temperature=0.4)
    summary = parse_json(summary_raw)
    if not isinstance(summary, dict):
        print(f"    {label} Warning: Summarizer failed — skipping")
        return None
    required_fields = ("headline", "one_liner", "detailed_summary")
    if not all(isinstance(summary.get(k), str) for k in required_fields):
        print(f"    {label} Warning: Summarizer missing fields — skipping")
        return None

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

    print(f"    {label} ✓ enriched")
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
    selected = state["selected"]
    print(f"\n=== Step 7: Enricher — processing {len(selected)} articles "
          f"(workers={ENRICH_WORKERS}) ===")

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

    results.sort(key=lambda x: x[0])
    final_articles = [r for _, r in results if r is not None]

    # Vocabulary dedup across all articles
    used_vocab_terms: set[str] = set(state.get("used_vocab_terms", []))
    for article in final_articles:
        deduped = []
        for v in article.get("vocabulary", []):
            term_key = v.get("term", "").lower().strip()
            if term_key and term_key not in used_vocab_terms:
                used_vocab_terms.add(term_key)
                deduped.append(v)
        article["vocabulary"] = deduped

    print(f"\n  > Enrichment complete: {len(final_articles)}/{len(selected)} articles")
    return {
        "final_articles": final_articles,
        "used_vocab_terms": list(used_vocab_terms),
        "output_json": final_articles,
        "selected": [],
    }


# ---------------------------------------------------------------------------
# Node 8: Build Output
# ---------------------------------------------------------------------------
def node_build_output(state: PipelineState) -> dict:
    final_articles = state["final_articles"]
    today_str = state["today_str"]
    print(f"\n=== Step 8: Build Output — {len(final_articles)} articles ===")
    save_archive(today_str, final_articles)
    cleanup_old_archives()
    archives = load_archives()
    print("  > Building HTML...")
    html = build_html(archives)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ Wrote {OUTPUT_FILE} with {len(archives)} days of archives")
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
        print(f"  ✗ Only {count} candidates — routing to fallback")
        return "fallback"
    return "continue"


def check_selected(state: PipelineState) -> str:
    count = len(state.get("selected", []))
    if count < MIN_ARTICLES:
        print(f"  ✗ Only {count} selected — routing to fallback")
        return "fallback"
    return "continue"


def check_enriched(state: PipelineState) -> str:
    count = len(state.get("final_articles", []))
    if count < MIN_ARTICLES:
        print(f"  ✗ Only {count} enriched — routing to fallback")
        return "fallback"
    return "continue"


# ---------------------------------------------------------------------------
# Graph Construction (LangGraph with ReAct curation loop)
# ---------------------------------------------------------------------------
def build_pipeline() -> StateGraph:
    workflow = StateGraph(PipelineState)

    workflow.add_node("fetch_sources",  node_fetch_sources)
    workflow.add_node("triage",         node_triage)
    workflow.add_node("investigate",    node_investigate)
    workflow.add_node("deep_score",     node_deep_score)
    workflow.add_node("reflect",        node_reflect)
    workflow.add_node("select",         node_select)
    workflow.add_node("enricher",       node_agent_enricher)
    workflow.add_node("build_output",   node_build_output)
    workflow.add_node("write_fallback", node_write_fallback)

    workflow.set_entry_point("fetch_sources")

    workflow.add_conditional_edges("fetch_sources", check_candidates, {
        "continue": "triage", "fallback": "write_fallback",
    })
    workflow.add_edge("triage", "investigate")
    workflow.add_edge("investigate", "deep_score")
    workflow.add_edge("deep_score", "reflect")

    # ReAct loop: reflect decides whether to loop or finalize
    workflow.add_conditional_edges("reflect", should_continue_react, {
        "investigate": "investigate",
        "select": "select",
    })

    workflow.add_conditional_edges("select", check_selected, {
        "continue": "enricher", "fallback": "write_fallback",
    })
    workflow.add_conditional_edges("enricher", check_enriched, {
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
# HTML Generation (unchanged from v5.0)
# ---------------------------------------------------------------------------
def build_html(archives: list[dict]) -> str:
    today_display = datetime.date.today().strftime("%B %d, %Y")
    archives_json = json.dumps(archives, ensure_ascii=False)
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
    AI Agent Daily Digest &mdash; ReAct multi-agent pipeline &bull; Gemini &bull; GitHub Actions
    <div class="pipeline-badge">v6.0 &middot; LangGraph ReAct &middot; 2-axis scoring &middot; Jina Reader &middot; 6-agent pipeline</div>
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
    print("  AI Agent Daily Digest — v6.0 (ReAct Curation Pipeline)")
    print("=" * 60)

    workflow = build_pipeline()
    app = workflow.compile()

    initial_state: PipelineState = {
        "candidates": [], "today_str": "", "status": "running",
        "recent_urls": [], "recent_headlines": [],
        "triage_scored": [],
        "shortlist": [], "investigated": [], "deep_scored": [],
        "react_iteration": 0, "react_action": "",
        "selected": [], "final_articles": [],
        "used_vocab_terms": [], "output_json": [],
    }

    final_state = None
    for state_snapshot in app.stream(initial_state, stream_mode="values"):
        final_state = state_snapshot

    print("\n" + "=" * 60)
    if final_state and final_state.get("status") == "success":
        articles = final_state.get("output_json", [])
        iterations = final_state.get("react_iteration", 0)
        print(f"  ✓ Pipeline SUCCESS — {len(articles)} articles published")
        print(f"  ✓ ReAct iterations: {iterations}")
        print(f"  ✓ HTML: {OUTPUT_FILE}")
        print(f"  ✓ Archive: {ARCHIVE_DIR}/")
    else:
        print("  ⚠ Pipeline completed with fallback")
    print("=" * 60)


if __name__ == "__main__":
    main()
