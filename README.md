# AI News for English Learners 📰

A fully automated, free-to-host MVP that curates one AI news article per day and turns it into a simple English lesson — vocabulary, example sentence, and discussion question.

**Stack:** Python · Google Gemini API · GitHub Actions · GitHub Pages

---

## Repository Structure

```
├── main.py                          # Fetches RSS → calls LLM → generates HTML
├── requirements.txt                 # Python dependencies
├── index.html                       # Auto-generated (don't edit by hand)
├── .github/workflows/
│   └── daily_update.yml             # Cron job: runs daily at 08:00 UTC
└── README.md
```

---

## Setup Instructions

### 1. Create the Repository

1. Go to **github.com → New repository**.
2. Name it whatever you like (e.g., `ai-news-esl`).
3. Make it **Public** (required for free GitHub Pages).
4. Upload all the files from this project, keeping the folder structure intact.

### 2. Add Your API Key as a Secret

1. In your repository, go to **Settings → Secrets and variables → Actions**.
2. Click **New repository secret**.
3. Set the **Name** to: `GEMINI_API_KEY`
4. Paste your Google Gemini API key as the **Value** (get one free at [aistudio.google.com](https://aistudio.google.com/apikey)).
5. Click **Add secret**.

> Your key is encrypted and never appears in logs or code.

### 3. Enable GitHub Pages

1. Go to **Settings → Pages**.
2. Under **Source**, select **Deploy from a branch**.
3. Set the branch to **main** and the folder to **/ (root)**.
4. Click **Save**.
5. After a minute your site will be live at: `https://<your-username>.github.io/<repo-name>/`

### 4. Test It

1. Go to **Actions** tab in your repo.
2. Select the **Daily AI News Update** workflow on the left.
3. Click the **Run workflow** dropdown button → **Run workflow**.
4. Wait ~30 seconds for it to finish, then check your GitHub Pages URL.

### 5. Customization

| What                          | Where                           |
|-------------------------------|---------------------------------|
| Change the schedule           | `daily_update.yml` → `cron`     |
| Use a different LLM model     | Set `GEMINI_MODEL` env variable |
| Add or swap RSS feeds         | `main.py` → `RSS_FEEDS` list   |
| Change the page design        | `main.py` → `build_html()`     |

The cron time `0 8 * * *` means 08:00 UTC every day. Use [crontab.guru](https://crontab.guru/) to adjust.

---

## How It Works

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ RSS Feed │────▶│ main.py  │────▶│ Gemini   │────▶│index.html│
│(TechCrunch)   │ (fetch)  │     │ (simplify)│    │ (static) │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                         │
                              GitHub Actions commits ─────┘
                              GitHub Pages serves it
```

Every day at 08:00 UTC, GitHub Actions runs `main.py`, which:
1. Pulls the latest article from RSS feeds
2. Sends it to the LLM with instructions to simplify it
3. Builds a clean, mobile-friendly HTML page
4. Commits and pushes `index.html` back to the repo

---

## Cost

- **GitHub Actions:** ~30 seconds/run × 30 days = well within the free tier (2,000 min/month)
- **Gemini API:** Free tier includes 15 requests/minute — more than enough for 1 daily call
- **Hosting:** Free via GitHub Pages

---

## License

MIT — do whatever you want with it.
