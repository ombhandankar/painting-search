import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from config import INGEST_MAX_SIDE, MODEL_NAME


class Embedder:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def _prepare_image(self, image: Image.Image, max_side: int) -> Image.Image:
        image = image.convert("RGB")
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize(
                (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
            )
        return image

    def embed(self, image: Image.Image, max_side: int = INGEST_MAX_SIDE) -> np.ndarray:
        image = self._prepare_image(image, max_side)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLS token embedding
            vector = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)
