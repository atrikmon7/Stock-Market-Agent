# India Market AI

A DealSize-style AI prototype for the Indian market. The app combines:

- A Google Sheet index of company fundamental documents.
- Yahoo Finance technical data via `yfinance`.
- A LangGraph backend orchestration layer.
- Gemini for text and image-aware responses.
- A chat interface with image/file attachment affordances.

## Setup

```powershell
cd backend
Copy-Item .env.example .env
```

Edit `backend/.env` and set:

```text
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MAX_OUTPUT_TOKENS=8192
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
```

Install and run the backend:

```powershell
cd ..
.\venv312test\Scripts\python.exe -m pip install -r backend\requirements.txt
cd backend
..\venv312test\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Install and run the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Notes

- Plain symbols are converted to NSE Yahoo symbols, for example `RELIANCE` becomes `RELIANCE.NS`.
- `NIFTY` maps to `^NSEI`; `BANKNIFTY` maps to `^NSEBANK`.
- The backend reads Google Sheets through `gspread`, preserving tab names, rows, columns, and metric-period alignment before sending structured JSON to the model.
- Set `GOOGLE_APPLICATION_CREDENTIALS` to the service account JSON file path for Sheets access.
- The model key is intentionally server-side only.
