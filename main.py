from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from PIL import Image
import uuid

app = FastAPI()

# init models (грузим один раз)
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1)

swapper = get_model('inswapper_128.onnx')

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def read_image(file: UploadFile):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def find_best_match(target_face, source_faces):
    sims = [
        np.dot(target_face.embedding, src.embedding)
        for src in source_faces
    ]
    return source_faces[int(np.argmax(sims))]


@app.post("/face-swap")
async def face_swap(
    generated: UploadFile = File(...),
    sources: list[UploadFile] = File(...)
):
    # читаем изображения
    generated_img = read_image(generated)

    source_images = [read_image(f) for f in sources]

    # детекция лиц
    gen_faces = face_app.get(generated_img)

    source_faces = []
    for img in source_images:
        faces = face_app.get(img)
        source_faces.extend(faces)

    if len(gen_faces) == 0 or len(source_faces) == 0:
        return {"error": "No faces found"}

    result = generated_img.copy()

    # свапаем каждое лицо
    for face in gen_faces:
        best_match = find_best_match(face, source_faces)

        result = swapper.get(
            result,
            face,
            best_match,
            paste_back=True
        )

    # сохраняем результат
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(TEMP_DIR, filename)

    cv2.imwrite(path, result)

    return FileResponse(path, media_type="image/jpeg")