import os
import socket

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    if not offline:
        try:
            socket.create_connection(("huggingface.co", 443), timeout=1)
        except OSError:
            offline = True

    kwargs.setdefault("local_files_only", offline)
    return loader.from_pretrained(model_name, **kwargs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "openai/clip-vit-base-patch32"

processor = _load_hf_model(CLIPProcessor, MODEL_NAME)
model = _load_hf_model(CLIPModel, MODEL_NAME)

model = model.to(device)
model.eval()

REAL_PROMPTS = [
    "a real photograph",
    "a real life photograph",
    "a photograph taken with a camera",
    "a photo of the real world",
    "a natural camera photograph",
    "a realistic photograph of a real person",
    "a realistic photograph of a real place",
    "a real world camera image",
    "a candid real life photo",
    "a photographic image of a real object"
]

ANIME_PROMPTS = [
    "an anime illustration",
    "a Japanese anime artwork",
    "a Japanese anime drawing",
    "a 2D anime character",
    "an anime screenshot",
    "a manga style illustration",
    "a Japanese animation frame",
    "digital anime artwork",
    "a cel shaded anime illustration",
    "a typical Japanese anime image"
]

THREE_D_PROMPTS = [
    "a 3D rendered image",
    "a computer generated 3D render",
    "a 3D CGI artwork",
    "a 3D modeled character",
    "a 3D computer animation frame",
    "a Blender style 3D render",
    "a realistic CGI render",
    "a digital 3D scene",
    "a 3D game character render",
    "a computer generated three dimensional image"
]

CARTOON_PROMPTS = [
    "a cartoon illustration",
    "a 2D cartoon drawing",
    "a western cartoon illustration",
    "a cartoon character drawing",
    "a colorful cartoon artwork",
    "a hand drawn cartoon",
    "a traditional cartoon animation frame",
    "a comic cartoon illustration",
    "a 2D animated cartoon frame",
    "a cartoon drawing that is not anime"
]

CLASS_PROMPTS = {
    "REAL LIFE": REAL_PROMPTS,
    "ANIME": ANIME_PROMPTS,
    "3D": THREE_D_PROMPTS,
    "CARTOON": CARTOON_PROMPTS
}

CLASS_NAMES = list(CLASS_PROMPTS.keys())

all_prompts = []

for class_name in CLASS_NAMES:
    all_prompts.extend(CLASS_PROMPTS[class_name])

text_inputs = processor(
    text=all_prompts,
    return_tensors="pt",
    padding=True
)

text_inputs = {
    key: value.to(device)
    for key, value in text_inputs.items()
}

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

    if not isinstance(text_features, torch.Tensor):
        if hasattr(text_features, "text_embeds"):
            text_features = text_features.text_embeds
        elif hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        else:
            text_features = text_features[0]

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

REAL_COUNT = len(REAL_PROMPTS)
ANIME_COUNT = len(ANIME_PROMPTS)
THREE_D_COUNT = len(THREE_D_PROMPTS)
CARTOON_COUNT = len(CARTOON_PROMPTS)

start = 0

real_text_features = text_features[
    start:start + REAL_COUNT
]

start += REAL_COUNT

anime_text_features = text_features[
    start:start + ANIME_COUNT
]

start += ANIME_COUNT

three_d_text_features = text_features[
    start:start + THREE_D_COUNT
]

start += THREE_D_COUNT

cartoon_text_features = text_features[
    start:start + CARTOON_COUNT
]

try:
    logit_scale = model.logit_scale.exp().item()
except Exception:
    logit_scale = 100.0


def get_class_score(image_features, class_features):
    similarities = (
        image_features @ class_features.T
    ).squeeze(0)

    return similarities.mean()


def classify_image(file_path):
    image = Image.open(file_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            else:
                image_features = image_features[0]

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        real_score = get_class_score(
            image_features,
            real_text_features
        )

        anime_score = get_class_score(
            image_features,
            anime_text_features
        )

        three_d_score = get_class_score(
            image_features,
            three_d_text_features
        )

        cartoon_score = get_class_score(
            image_features,
            cartoon_text_features
        )

        class_logits = torch.stack([
            real_score,
            anime_score,
            three_d_score,
            cartoon_score
        ]) * logit_scale

        probabilities = torch.softmax(
            class_logits,
            dim=0
        )

        prob_real = probabilities[0].item()
        prob_anime = probabilities[1].item()
        prob_3d = probabilities[2].item()
        prob_cartoon = probabilities[3].item()

    results = {
        "real life": prob_real,
        "anime": prob_anime,
        "3D": prob_3d,
        "cartoon": prob_cartoon
    }

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_class = sorted_results[0][0]
    best_probability = sorted_results[0][1]

    return {
        "best_class": best_class,
        "best_probability": best_probability,
        "results": results,
        "sorted_results": sorted_results,
        "device": str(device)
    }
