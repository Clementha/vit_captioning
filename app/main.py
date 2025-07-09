# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
from pathlib import Path

from generate import CaptionGenerator

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ✅ Landing page at `/`
@app.get("/", response_class=HTMLResponse)
async def landing():
    return Path("app/static/landing.html").read_text()

# ✅ Captioning page at `/captioning`
@app.get("/captioning", response_class=HTMLResponse)
async def captioning():
    return Path("app/static/captioning/index.html").read_text()

# ✅ Example: Project 2 placeholder
@app.get("/project2", response_class=HTMLResponse)
async def project2():
    return "<h1>Coming Soon: Project 2</h1>"

# ✅ Caption generation endpoint for captioning app
# Keep the path consistent with your JS fetch()!
caption_generator = CaptionGenerator(
    model_type="CLIPEncoder",
    checkpoint_path="./artifacts/CLIPEncoder_40epochs_unfreeze12.pth"
)

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    captions = caption_generator.generate_caption(temp_file)
    return captions