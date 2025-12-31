from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from markdown import markdown
import os
from utils import extract_text_from_pdf, calculate_match_score

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Setup Gemini API
API_KEY = "YOUR API KEY"

client = genai.Client(api_key=API_KEY)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_cv(
    request: Request,
    cv_file: UploadFile = File(...),
    job_desc: str = Form(...)
):
    content = await cv_file.read()
    cv_text = extract_text_from_pdf(content)

    if not cv_text:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Gagal membaca PDF. Pastikan file tidak rusak."
        })

    match_score = calculate_match_score(cv_text, job_desc)
    ai_feedback = "Gagal menghubungi AI."
    try:
        prompt = f"""
        Bertindaklah sebagai HR Manager Senior.
        Review CV kandidat ini berdasarkan Job Description.
        
        Skor kecocokan teknis (ATS Score): {match_score}%
        
        Job Desc:
        {job_desc}
        
        CV Kandidat:
        {cv_text}
        
        Berikan:
        1. Analisis kenapa skornya {match_score}%.
        2. 3 Skill penting yang hilang.
        3. Saran perbaikan singkat.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        
        ai_feedback = markdown(response.text)
        
    except Exception as e:
        print(f"Error Gemini: {e}")
        ai_feedback = f"Maaf, terjadi error pada AI: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "score": match_score,
        "feedback": ai_feedback,
        "filename": cv_file.filename
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)