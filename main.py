from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import uuid

app = FastAPI()

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(1024, 1024))

swapper = get_model('inswapper_128.onnx', providers=['CPUExecutionProvider'])

TEMP_DIR = "temp"
MAX_SOURCE_FACES = 30

os.makedirs(TEMP_DIR, exist_ok=True)


def read_image(file: UploadFile):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(face_a, face_b):
    emb_a = normalize_embedding(face_a.embedding)
    emb_b = normalize_embedding(face_b.embedding)
    return float(np.dot(emb_a, emb_b))


def face_area(face):
    x1, y1, x2, y2 = face.bbox
    return float((x2 - x1) * (y2 - y1))


def collect_source_faces(source_images):
    source_faces = []

    for img in source_images:
        if img is None:
            continue

        faces = face_app.get(img)
        source_faces.extend(faces)

    source_faces.sort(key=face_area, reverse=True)
    return source_faces[:MAX_SOURCE_FACES]


def find_best_match_index(target_face, source_faces):
    similarities = [
        cosine_similarity(target_face, source_face)
        for source_face in source_faces
    ]
    return int(np.argmax(similarities))


@app.get("/")
async def root():
    return "Face Swap API is running. Use POST /face-swap to swap faces."


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is healthy"}


@app.post("/face-swap")
async def face_swap(
    generated: UploadFile = File(...),
    sources: list[UploadFile] = File(...)
):
    generated_img = read_image(generated)
    if generated_img is None:
        raise HTTPException(status_code=422, detail="Generated image is invalid")

    source_images = [read_image(file) for file in sources]

    gen_faces = face_app.get(generated_img)
    source_faces = collect_source_faces(source_images)

    if len(gen_faces) == 0:
        raise HTTPException(status_code=422, detail="No faces found in generated image")

    if len(source_faces) == 0:
        raise HTTPException(status_code=422, detail="No faces found in source images")

    result = generated_img.copy()
    available_source_faces = list(source_faces)

    for gen_face in gen_faces:
        candidates = available_source_faces if len(available_source_faces) > 0 else source_faces
        best_match_index = find_best_match_index(gen_face, candidates)
        best_match = candidates[best_match_index]

        if len(available_source_faces) > 0:
            available_source_faces.pop(best_match_index)

        result = swapper.get(
            result,
            gen_face,
            best_match,
            paste_back=True
        )

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(TEMP_DIR, filename)

    cv2.imwrite(path, result)

    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={
            "X-Source-Face-Count": str(len(source_faces)),
            "X-Generated-Face-Count": str(len(gen_faces))
        }
    )
