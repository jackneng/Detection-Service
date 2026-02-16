import base64
import io
import tempfile
import uuid
from typing import List

from fastapi import FastAPI
from nudenet import NudeDetector
from PIL import Image
from pydantic import BaseModel
from transformers import pipeline

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LIKELIHOOD_THRESHOLDS = [
    (0.2, "VERY_UNLIKELY"),
    (0.4, "UNLIKELY"),
    (0.6, "POSSIBLE"),
    (0.8, "LIKELY"),
]
MODERATION_LABELS = [
    "normal message",
    "fraud or scam message",
    "violent message",
    "sexual content",
]
MODERATION_THRESHOLD = 0.75
CATEGORY_KEYWORDS = {"fraud": "fraud", "violence": "violent", "sexual": "sexual"}

# -----------------------------------------------------------------------------
# App & models
# -----------------------------------------------------------------------------

app = FastAPI()
detector = NudeDetector()
text_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
)


# ----------- Google Vision request/response models -----------


class Feature(BaseModel):
    type: str


class ImageContent(BaseModel):
    content: str  # base64 image


class VisionRequestItem(BaseModel):
    image: ImageContent
    features: List[Feature]


class VisionRequest(BaseModel):
    requests: List[VisionRequestItem]


class ModerationRequest(BaseModel):
    model: str
    input: str


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def score_to_likelihood(score: float) -> str:
    for threshold, label in LIKELIHOOD_THRESHOLDS:
        if score < threshold:
            return label
    return "VERY_LIKELY"


def _run_image_detection(image_bytes: bytes) -> float:
    with io.BytesIO(image_bytes) as buf:
        image = Image.open(buf).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        image.save(tmp.name)
        detections = detector.detect(tmp.name)
    if not detections:
        return 0.0
    return max(d["score"] for d in detections)


def _build_moderation_result(labels: List[str], scores: List[float]) -> tuple[dict, dict, bool]:
    categories = {key: False for key in CATEGORY_KEYWORDS}
    category_scores = {key: 0.0 for key in CATEGORY_KEYWORDS}

    for label, score in zip(labels, scores):
        for category, keyword in CATEGORY_KEYWORDS.items():
            if keyword in label.lower():
                category_scores[category] = score
                categories[category] = score > MODERATION_THRESHOLD
                break

    flagged = any(categories.values())
    return categories, category_scores, flagged


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.post("/v1/images:annotate")
async def annotate_image(req: VisionRequest) -> dict:
    item = req.requests[0]
    image_bytes = base64.b64decode(item.image.content)
    max_score = _run_image_detection(image_bytes)
    likelihood = score_to_likelihood(max_score)

    return {
        "responses": [
            {
                "safeSearchAnnotation": {
                    "adult": likelihood,
                    "violence": "VERY_UNLIKELY",
                    "racy": likelihood,
                }
            }
        ],
    }


@app.post("/v1/moderations")
async def moderate(req: ModerationRequest) -> dict:
    result = text_classifier(req.input, MODERATION_LABELS)
    categories, category_scores, flagged = _build_moderation_result(
        result["labels"], result["scores"]
    )

    return {
        "id": f"mod-{uuid.uuid4()}",
        "model": req.model,
        "results": [
            {
                "flagged": flagged,
                "categories": categories,
                "category_scores": category_scores,
            }
        ],
    }
