# AI Agent Daily Digest

A fully automated, free-to-host news digest that curates 3-5 AI agent articles daily — with English vocabulary lessons, detailed summaries, and a 30-day browsable archive.

**Stack:** Python · Google Gemini API (free tier) · GitHub Actions · GitHub Pages

---

## Features

- **3-5 curated articles daily** filtered for AI agent relevance (releases, technical deep-dives, use cases)
- **Expandable cards** — headline + one-liner on the surface, click to reveal full summary, vocabulary, and discussion questions
- **English learning** — 3 vocabulary words per article with definitions and example sentences (English-only, dictionary style)
- **30-day archive** — browse past digests with date navigation and archive grid
- **Dark-themed UI** — clean, mobile-friendly, editorial design
- **Category tagging** — release / technical / use-case / industry / research

## Repository structure

```
├── main.py                          # RSS → Gemini → HTML pipeline
├── requirements.txt                 # Python dependencies
├── index.html                       # Auto-generated (don't edit)
├── archives/                        # Daily JSON archives (auto-managed)
│   ├── 2026-03-24.json
│   ├── 2026-03-23.json
│   └── ...
├── .github/workflows/
│   └── daily_update.yml             # Runs daily at 08:00 JST
└── README.md
```

## Setup

1. **Create a public GitHub repository** and upload all files
2. **Add your Gemini API key**: Settings → Secrets → Actions → `GEMINI_API_KEY` (get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))
3. **Enable GitHub Pages**: Settings → Pages → Deploy from branch → main / root
4. **Test**: Actions tab → Daily AI Agent Digest → Run workflow

Your site will be live at `https://<username>.github.io/<repo>/`

## Cost

- **Gemini API**: Free tier (15 RPM) — this uses 1 call/day
- **GitHub Actions**: ~30 sec/run — well within free tier
- **Hosting**: Free via GitHub Pages

**Total: $0/month**

## License

MIT
