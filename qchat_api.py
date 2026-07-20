import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from xgboost_random_forest_model import XGBoostRandomForestModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_random_forest_model.pkl")
_model: XGBoostRandomForestModel | None = None


def _get_model() -> XGBoostRandomForestModel:
    global _model
    if _model is None:
        m = XGBoostRandomForestModel()
        m.load_model(MODEL_PATH)
        _model = m
    return _model


def preload():
    """Eagerly load the model. Called from main.py's startup."""
    _get_model()


router = APIRouter(prefix="/qchat", tags=["qchat"])


def _qchat_options(labels: list[str]) -> list[dict]:
    """Five options in strict A → B → C → D → E order (paper form)."""
    letters = ["A", "B", "C", "D", "E"]
    return [{"letter": letters[i], "label": labels[i]} for i in range(5)]


_QCHAT_ALWAYS_USUALLY = [
    "Always",
    "Usually",
    "Sometimes",
    "Rarely",
    "Never",
]
_QCHAT_EYE_CONTACT = [
    "Very easy",
    "Quite easy",
    "Quite difficult",
    "Very difficult",
    "Impossible",
]
_QCHAT_FREQ = [
    "Many times a day",
    "A few times a day",
    "A few times a week",
    "Less than once a week",
    "Never",
]
_QCHAT_FIRST_WORDS = [
    "Very typical",
    "Quite typical",
    "Slightly unusual",
    "Very unusual",
    "My child doesn't speak",
]

SCREENING_QUESTIONS = [
    {
        "id": "A1",
        "text": "Does your child look at you when you call his/her name?",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_ALWAYS_USUALLY),
    },
    {
        "id": "A2",
        "text": "How easy is it for you to get eye contact with your child?",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_EYE_CONTACT),
    },
    {
        "id": "A3",
        "text": "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_FREQ),
    },
    {
        "id": "A4",
        "text": "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_FREQ),
    },
    {
        "id": "A5",
        "text": "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_FREQ),
    },
    {
        "id": "A6",
        "text": "Does your child follow where you're looking?",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_FREQ),
    },
    {
        "id": "A7",
        "text": "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_ALWAYS_USUALLY),
    },
    {
        "id": "A8",
        "text": "Would you describe your child's first words as:",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_FIRST_WORDS),
    },
    {
        "id": "A9",
        "text": "Does your child use simple gestures? (e.g. wave goodbye)",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_FREQ),
    },
    {
        "id": "A10",
        "text": "Does your child stare at nothing with no apparent purpose?",
        "type": "qchat",
        "options": _qchat_options(_QCHAT_FREQ),
    },
    {
        "id": "age_months",
        "text": "What is the child's age in months?",
        "type": "numeric",
        "options": None,
    },
    {
        "id": "gender",
        "text": "What is the child's gender?",
        "type": "choice",
        "options": [
            {"value": "male", "label": "Male"},
            {"value": "female", "label": "Female"},
        ],
    },
    {
        "id": "jaundice",
        "text": "Was the child born with jaundice?",
        "type": "choice",
        "options": [
            {"value": "yes", "label": "Yes"},
            {"value": "no", "label": "No"},
        ],
    },
    {
        "id": "family_asd",
        "text": "Does any immediate family member have a history of ASD?",
        "type": "choice",
        "options": [
            {"value": "yes", "label": "Yes"},
            {"value": "no", "label": "No"},
        ],
    },
]


def qchat_item_to_binary(item_index_1_to_10: int, letter: str) -> int:
    """Map paper column A–E to Q-CHAT binary (0/1) per official scoring."""
    ch = letter.strip().upper()
    if len(ch) != 1 or ch not in "ABCDE":
        raise ValueError(f"Invalid Q-CHAT letter: {letter!r}")
    idx = ord(ch) - ord("A")
    if 1 <= item_index_1_to_10 <= 9:
        return 1 if idx >= 2 else 0
    if item_index_1_to_10 == 10:
        return 1 if idx <= 2 else 0
    raise ValueError(f"Invalid item index: {item_index_1_to_10}")


_QCHAT_FIELDS = tuple(f"A{i}" for i in range(1, 11))


class ScreeningAnswers(BaseModel):
    A1: str = Field(..., description="Paper column A–E for Q-CHAT item 1")
    A2: str = Field(..., description="Paper column A–E for Q-CHAT item 2")
    A3: str = Field(..., description="Paper column A–E for Q-CHAT item 3")
    A4: str = Field(..., description="Paper column A–E for Q-CHAT item 4")
    A5: str = Field(..., description="Paper column A–E for Q-CHAT item 5")
    A6: str = Field(..., description="Paper column A–E for Q-CHAT item 6")
    A7: str = Field(..., description="Paper column A–E for Q-CHAT item 7")
    A8: str = Field(..., description="Paper column A–E for Q-CHAT item 8")
    A9: str = Field(..., description="Paper column A–E for Q-CHAT item 9")
    A10: str = Field(..., description="Paper column A–E for Q-CHAT item 10")
    age_months: float = Field(..., gt=0, le=100)
    gender: str = Field(..., pattern=r"^(male|female)$")
    jaundice: str = Field(..., pattern=r"^(yes|no)$")
    family_asd: str = Field(..., pattern=r"^(yes|no)$")

    @field_validator(*_QCHAT_FIELDS, mode="before")
    @classmethod
    def normalize_qchat_letter(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        raise TypeError("Each A1–A10 answer must be a string letter A–E")

    @field_validator(*_QCHAT_FIELDS)
    @classmethod
    def validate_qchat_letter(cls, v: str) -> str:
        if len(v) != 1 or v not in "ABCDE":
            raise ValueError("Each A1–A10 must be a single letter A, B, C, D, or E")
        return v


@router.get("/questions")
def get_questions():
    return {
        "scoring_note": (
            "Official Q-CHAT-10: items 1–9 score 1 for C, D, or E; item 10 scores 1 for A, B, or C. "
            "Options are listed in paper order A through E. POST /classify accepts each answer as a letter A–E."
        ),
        "questions": SCREENING_QUESTIONS,
    }


def _classify_sync(answers: ScreeningAnswers):
    model = _get_model()
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    payload = answers.model_dump()
    row = {
        f"A{i}": qchat_item_to_binary(i, payload[f"A{i}"]) for i in range(1, 11)
    }
    row["age_months"] = payload["age_months"]
    row["gender"] = payload["gender"]
    row["jaundice"] = payload["jaundice"]
    row["family_asd"] = payload["family_asd"]
    row["asd_class"] = 0  # dummy column required by the scaler
    df = pd.DataFrame([row])

    try:
        processed = model.preprocess_data(df, is_training=False)
        features = model.prepare_features(processed)
        features = features.drop("asd_class", axis=1, errors="ignore")

        prediction = int(model.ensemble_model.predict(features)[0])
        probabilities = model.ensemble_model.predict_proba(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    no_asd_prob = round(float(probabilities[0]), 4)
    asd_prob = round(float(probabilities[1]), 4)
    confidence = round(max(no_asd_prob, asd_prob) * 100, 2)

    return {
        "classification": prediction,
        "label": "ASD traits detected" if prediction == 1 else "No ASD traits",
        "confidence": confidence,
        "risk_level": "High Risk" if prediction == 1 else "Low Risk",
        "probabilities": {"no_asd": no_asd_prob, "asd": asd_prob},
    }


@router.post("/classify")
async def classify(answers: ScreeningAnswers):
    return await run_in_threadpool(_classify_sync, answers)
