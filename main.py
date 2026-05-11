"""
DogVision – Backend de prueba (mock)
Imita exactamente la API que consume el frontend Angular.

Endpoints:
  GET  /health    → estado del modelo
  POST /predict   → analiza imagen y retorna predicciones de raza
  GET  /classes   → lista de razas disponibles
  GET  /metrics   → métricas del modelo
"""

import random
import time
import base64
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="DogVision Mock API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
# Permite peticiones desde el frontend Angular (localhost:4200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Datos de prueba ───────────────────────────────────────────────────────────

BREEDS = [
    "n02085620-Chihuahua",
    "n02086240-Shih-Tzu",
    "n02086646-Blenheim_spaniel",
    "n02088238-beagle",
    "n02088364-beagle",
    "n02089973-English_foxhound",
    "n02090379-redbone",
    "n02090622-borzoi",
    "n02091032-Italian_greyhound",
    "n02091467-Norwegian_elkhound",
    "n02093256-Staffordshire_bullterrier",
    "n02093428-American_Staffordshire_terrier",
    "n02095314-wire-haired_fox_terrier",
    "n02096294-Australian_terrier",
    "n02096437-Dandie_Dinmont",
    "n02097298-Scotch_terrier",
    "n02099601-golden_retriever",
    "n02099712-Labrador_retriever",
    "n02105505-komondor",
    "n02106550-Rottweiler",
    "n02107574-Greater_Swiss_Mountain_dog",
    "n02108000-EntleBucher",
    "n02108915-French_bulldog",
    "n02109525-Saint_Bernard",
    "n02110185-Siberian_husky",
    "n02110627-affenpinscher",
    "n02111129-Leonberg",
    "n02111500-Great_Pyrenees",
    "n02112018-Pomeranian",
    "n02113023-Pembroke",
    "n02113624-toy_poodle",
    "n02113712-miniature_poodle",
    "n02113799-standard_poodle",
    "n02114367-timber_wolf",
    "n02115641-dingo",
    "n02116738-African_hunting_dog",
]


# ── Modelos Pydantic ──────────────────────────────────────────────────────────

class BreedPrediction(BaseModel):
    breed: str
    confidence: float

class PredictResponse(BaseModel):
    predictions: list[BreedPrediction]
    top_breed: str
    top_confidence: float
    gradcam_base64: Optional[str]
    inference_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    device: str
    model_exists: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_gradcam_png() -> str:
    """Genera un PNG mínimo en base64 que simula un mapa GradCAM."""
    try:
        from PIL import Image, ImageDraw
        import numpy as np

        w, h = 224, 224
        img = Image.new("RGB", (w, h), (20, 20, 80))
        draw = ImageDraw.Draw(img)
        # Mancha de calor simulada
        for r in range(80, 0, -10):
            alpha = int(255 * (1 - r / 80))
            color = (255, alpha, 0)
            draw.ellipse(
                [w // 2 - r, h // 2 - r, w // 2 + r, h // 2 + r],
                fill=color,
            )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Si Pillow no está, retorna un PNG de 1×1 transparente
        TINY_PNG = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        return TINY_PNG


def generate_predictions(top_breed: str) -> list[BreedPrediction]:
    """Genera 5 predicciones coherentes con confidencias que suman ~1."""
    other_breeds = [b for b in BREEDS if b != top_breed]
    picked = random.sample(other_breeds, 4)

    top_conf = random.uniform(0.55, 0.92)
    remaining = 1.0 - top_conf
    confs = sorted([random.random() for _ in range(3)], reverse=True)
    total = sum(confs)
    confs = [c / total * remaining for c in confs]
    confs.append(remaining - sum(confs))

    predictions = [BreedPrediction(breed=top_breed, confidence=top_conf)]
    for breed, conf in zip(picked, confs):
        predictions.append(BreedPrediction(breed=breed, confidence=max(0.001, conf)))

    return predictions


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=True,
        num_classes=len(BREEDS),
        device="cpu",
        model_exists=True,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # Validar que es imagen
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    await file.read()  # simulamos leerla

    # Simular latencia de inferencia
    start = time.time()
    time.sleep(random.uniform(0.3, 0.8))
    inference_ms = (time.time() - start) * 1000

    top_breed = random.choice(BREEDS)
    predictions = generate_predictions(top_breed)

    # GradCAM simulado (50 % de probabilidad de incluirlo)
    gradcam = make_fake_gradcam_png() if random.random() > 0.5 else None

    return PredictResponse(
        predictions=predictions,
        top_breed=top_breed,
        top_confidence=predictions[0].confidence,
        gradcam_base64=gradcam,
        inference_ms=round(inference_ms, 1),
    )


@app.get("/classes")
def get_classes():
    return {"classes": BREEDS, "count": len(BREEDS)}


@app.get("/metrics")
def get_metrics():
    return {
        "accuracy_top1": 0.834,
        "accuracy_top5": 0.961,
        "total_predictions": random.randint(200, 5000),
        "avg_inference_ms": round(random.uniform(180, 450), 1),
        "model_name": "DogVisionMock-v1",
        "num_classes": len(BREEDS),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
