# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
from pathlib import Path

from generate import CaptionGenerator

# --------------------------------------------
# 1️⃣ Create FastAPI app & serve static files
# --------------------------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")


# --------------------------------------------
# 2️⃣ Load your model ONCE when the server starts
# --------------------------------------------
# You can change model_type and checkpoint_path here!
caption_generator = CaptionGenerator(
    model_type="CLIPEncoder",               # or "CLIPEncoder"
    checkpoint_path="./artifacts/CLIPEncoder_40epochs_unfreeze12.pth"  # adjust path as needed
)


# --------------------------------------------
# 3️⃣ Serve your HTML page
# --------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("app/static/index.html").read_text()


# --------------------------------------------
# 4️⃣ Endpoint: POST an image, get captions
# --------------------------------------------
@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    # Save uploaded file to a temporary path
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate captions (returns dict)
    captions = caption_generator.generate_caption(temp_file)
    #print(f"Generated captions: {captions}")
    return captions  # FastAPI auto-converts to JSON