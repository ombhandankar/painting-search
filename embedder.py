import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from config import INGEST_MAX_SIDE, MODEL_NAME, TORCH_NUM_THREADS


class Embedder:
    def __init__(self):
        torch.set_num_threads(TORCH_NUM_THREADS)
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

    def embed_batch(
        self, images: list[Image.Image], max_side: int = INGEST_MAX_SIDE
    ) -> np.ndarray:
        prepared = [self._prepare_image(image, max_side) for image in images]
        inputs = self.processor(images=prepared, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, np.finfo(np.float32).eps)
        return vectors.astype(np.float32)

    def embed(self, image: Image.Image, max_side: int = INGEST_MAX_SIDE) -> np.ndarray:
        return self.embed_batch([image], max_side=max_side)[0]
