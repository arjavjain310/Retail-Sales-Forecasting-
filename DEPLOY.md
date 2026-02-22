# Deploy Retail Mart Dashboard

## Push to GitHub

Repo: [https://github.com/arjavjain310/Retail-Sales-Forecasting-](https://github.com/arjavjain310/Retail-Sales-Forecasting-)

```bash
cd "Retail Sales Forecasting"
git init
git add .
git commit -m "Retail Sales Forecasting & Dashboard (INR, Streamlit)"
git branch -M main
git remote add origin https://github.com/arjavjain310/Retail-Sales-Forecasting-.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select repo: **arjavjain310/Retail-Sales-Forecasting-**.
4. Branch: **main**.
5. Main file path: **app.py**.
6. Click **Deploy**. Streamlit will use `requirements.txt` and `runtime.txt` from the repo.
7. The app will download data from the dataset URL on first run (no secrets needed).

Your live dashboard URL will be:  
`https://<your-app-name>.streamlit.app`
