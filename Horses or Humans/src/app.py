import io

import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from frontend.cam_utils import predict_and_generate_cam
from PIL import Image
from starlette.responses import Response

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Upload an image (Horse or Human):</h2>
            <form action="/upload/" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
            </form>
        </body>
    </html>
    """


@app.post("/upload/")
def upload(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result_image = predict_and_generate_cam(image)
    _, image_encoded = cv2.imencode(".png", result_image)

    return Response(content=image_encoded.tobytes(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="127.0.0.1", port=8_000, reload=True)
