from __future__ import annotations

import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import run_training_pipeline

app = FastAPI(title="Education Analytics MLOps Platform")


class PredictRequest(BaseModel):
    gender: str
    race_ethnicity: str = Field(..., alias="race_ethnicity")
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

    class Config:
        populate_by_name = True


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"status": "ok", "message": "Student performance service"}


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, Any]:
    """Return a single prediction using the persisted model artifacts."""
    try:
        pipeline = PredictPipeline()
        data = payload.model_dump(by_alias=True)
        prediction = pipeline.predict_from_dict(data)
        return {"prediction": float(prediction)}
    except Exception as exc: 
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train")
def train() -> Dict[str, Any]:
    """Trigger training and return artifact metadata."""
    try:
        return run_training_pipeline()
    except Exception as exc: 
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
        """Minimal HTML UI for interactive predictions."""
        return """
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"UTF-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>Student Performance Predictor</title>
            <style>
                body { font-family: Arial, sans-serif; background:#0d1b2a; color:#e0e6ed; margin:0; padding:24px; display:flex; justify-content:center; }
                .card { background:#14213d; border:1px solid #1f2f4a; border-radius:10px; padding:24px; max-width:540px; width:100%; box-shadow:0 12px 30px rgba(0,0,0,0.35); }
                h1 { margin-top:0; letter-spacing:0.4px; }
                label { display:block; margin:12px 0 6px; font-weight:600; }
                input, select { width:100%; padding:10px; border-radius:6px; border:1px solid #324a6b; background:#0b1323; color:#e0e6ed; }
                button { margin-top:16px; width:100%; padding:12px; background:#fca311; color:#0d1b2a; border:none; border-radius:8px; font-weight:700; cursor:pointer; }
                button:hover { background:#ffb347; }
                .result { margin-top:18px; padding:12px; background:#0b1323; border:1px solid #324a6b; border-radius:8px; min-height:40px; }
                .error { color:#ff6b6b; font-weight:700; }
                .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
                @media (max-width:640px) { .row { grid-template-columns:1fr; } }
            </style>
        </head>
        <body>
            <div class=\"card\">
                <div style=\"text-align:center; margin-bottom:24px; padding-bottom:16px; border-bottom:2px solid #324a6b;\">
                    <h2 style=\"margin:0 0 8px; color:#fca311; font-size:18px; letter-spacing:1px;\">Education Analytics MLOps Platform</h2>
                    <p style=\"margin:0; color:#a0afc0; font-size:14px;\">Predictive Analytics & Model Management</p>
                </div>
                <form id=\"predict-form\">
                    <label>Gender</label>
                    <select name=\"gender\" required>
                        <option value=\"female\">Female</option>
                        <option value=\"male\">Male</option>
                    </select>

                    <label>Race / Ethnicity</label>
                    <select name=\"race_ethnicity\" required>
                        <option value=\"group A\">Group A</option>
                        <option value=\"group B\">Group B</option>
                        <option value=\"group C\">Group C</option>
                        <option value=\"group D\">Group D</option>
                        <option value=\"group E\">Group E</option>
                    </select>

                    <label>Parental Level of Education</label>
                    <select name=\"parental_level_of_education\" required>
                        <option value=\"bachelor's degree\">Bachelor's degree</option>
                        <option value=\"master's degree\">Master's degree</option>
                        <option value=\"associate's degree\">Associate's degree</option>
                        <option value=\"some college\">Some college</option>
                        <option value=\"high school\">High school</option>
                        <option value=\"some high school\">Some high school</option>
                    </select>

                    <label>Lunch</label>
                    <select name=\"lunch\" required>
                        <option value=\"standard\">Standard</option>
                        <option value=\"free/reduced\">Free/Reduced</option>
                    </select>

                    <label>Test Preparation Course</label>
                    <select name=\"test_preparation_course\" required>
                        <option value=\"none\">None</option>
                        <option value=\"completed\">Completed</option>
                    </select>

                    <div class=\"row\">
                        <div>
                            <label>Reading Score</label>
                            <input type=\"number\" step=\"0.01\" name=\"reading_score\" required />
                        </div>
                        <div>
                            <label>Writing Score</label>
                            <input type=\"number\" step=\"0.01\" name=\"writing_score\" required />
                        </div>
                    </div>

                    <button type=\"submit\">Predict</button>
                </form>
                <div class=\"result\" id=\"result\">Prediction will appear here.</div>
            </div>

            <script>
                const form = document.getElementById('predict-form');
                const resultBox = document.getElementById('result');

                form.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    resultBox.textContent = 'Predicting...';
                    const formData = new FormData(form);
                    const payload = Object.fromEntries(formData.entries());
                    payload.reading_score = Number(payload.reading_score);
                    payload.writing_score = Number(payload.writing_score);
                    try {
                        const res = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload),
                        });
                        if (!res.ok) {
                            const text = await res.text();
                            throw new Error(text || 'Request failed');
                        }
                        const data = await res.json();
                        resultBox.textContent = `Predicted math score: ${data.prediction.toFixed(2)}`;
                    } catch (err) {
                        resultBox.innerHTML = `<span class=\"error\">Error: ${err}</span>`;
                    }
                });
            </script>
        </body>
        </html>
        """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
