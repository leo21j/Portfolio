import numpy as np
import requests

from scripts.logger import get_logger

log = get_logger("embedder")


class OllamaEmbedder:
    """Embeds text through a local Ollama server.

    All returned vectors are L2-normalized. Downstream code depends on this:
    the retriever searches a ``faiss.IndexFlatIP`` (inner product) index, which
    only equals cosine similarity on unit vectors. Ollama's embedding models
    already return normalized output, so this is a no-op in practice; it is
    applied anyway so swapping ``embed_model`` cannot silently turn retrieval
    into a magnitude-biased dot product.
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:8b",
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        timeout: int = 1800,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self._dim: int | None = None

        log.info(f"Connecting to Ollama embedder ({model}) at {base_url}...")
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            # Determine embedding dimension with a single real call
            dim = self._embed_batch(["dim_probe"])[0].shape[0]
            self._dim = dim
            log.success(f"OllamaEmbedder ready — model={model}, dim={dim}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot initialize Ollama embedder at {base_url}. "
                f"Is 'ollama serve' running and '{model}' pulled? Error: {e}"
            ) from e

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        all_vecs = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if show_progress:
                batch_num = i // self.batch_size + 1
                log.info(f"Embedding batch {batch_num}/{n_batches} ({len(batch)} texts)...")
            vecs = self._embed_batch(batch)
            all_vecs.append(vecs)

        return np.vstack(all_vecs).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts, "keep_alive": -1}

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.ConnectionError as e:
            raise RuntimeError(f"Ollama not reachable at {url}") from e
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama /api/embed returned error: {e}") from e

        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama response missing 'embeddings' key: {data}")

        vecs = np.array(embeddings, dtype=np.float32)  # (N, dim)
        return self._l2_normalize(vecs)

    @staticmethod
    def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalization so dot product == cosine similarity."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        return vecs / norms
