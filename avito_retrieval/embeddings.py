"""Local embedding inference with the downloaded FRIDA model."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


PromptName = Literal["paraphrase", "search_query", "search_document"]


class FridaEmbedder:
    """Run FRIDA locally through Sentence Transformers without network access."""

    def __init__(
        self,
        model_path: str | Path = "models/frida",
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device or _default_device()
        self.batch_size = batch_size
        self._model = None

    def encode(self, texts: list[str], prompt_name: PromptName) -> np.ndarray:
        """Return L2-normalized float32 vectors for one FRIDA retrieval task."""
        if not texts:
            return np.empty((0, 1536), dtype=np.float32)

        embeddings = self._load_model().encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    f"FRIDA model is not available at {self.model_path}. "
                    "Download ai-forever/FRIDA into this directory first."
                )
            dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
            self._model = SentenceTransformer(
                str(self.model_path),
                device=self.device,
                local_files_only=True,
                model_kwargs={"torch_dtype": dtype},
            )
        return self._model


def _default_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
