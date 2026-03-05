import numpy as np
import pandas as pd
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")


class SemanticAnalyzer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers is required for semantic analysis")
        self.model = SentenceTransformer(model_name)
        self._embedding_cache: dict[str, np.ndarray] = {}

    def compute_embeddings(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])

        texts_to_encode = []
        indices_to_encode = []
        embeddings = [None] * len(texts)

        for i, text in enumerate(texts):
            if use_cache and text in self._embedding_cache:
                embeddings[i] = self._embedding_cache[text]
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)

        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode, show_progress_bar=False)
            for idx, emb, text in zip(indices_to_encode, new_embeddings, texts_to_encode):
                embeddings[idx] = emb
                if use_cache:
                    self._embedding_cache[text] = emb

        return np.array(embeddings)

    def semantic_variance(self, embeddings: np.ndarray) -> float:
        if len(embeddings) == 0:
            return 0.0
        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return float(distances.var())

    def semantic_spread(self, embeddings: np.ndarray) -> float:
        if len(embeddings) == 0:
            return 0.0
        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return float(distances.mean())

    def pairwise_cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.array([[]])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        return normalized @ normalized.T

    def mean_pairwise_similarity(self, embeddings: np.ndarray) -> float:
        if len(embeddings) < 2:
            return 1.0
        sim_matrix = self.pairwise_cosine_similarity(embeddings)
        n = len(embeddings)
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        return float(upper_tri.mean())

    def polarization_index(self, embeddings: np.ndarray) -> float:
        if len(embeddings) < 2:
            return 0.0
        mean_sim = self.mean_pairwise_similarity(embeddings)
        return 1.0 - mean_sim

    def clear_cache(self):
        self._embedding_cache.clear()


def compute_semantic_metrics_over_time(
    df: pd.DataFrame,
    analyzer: Optional[SemanticAnalyzer] = None,
    reason_column: str = "reason"
) -> pd.DataFrame:
    if analyzer is None:
        analyzer = SemanticAnalyzer()

    results = []

    grouped = df.groupby(["param_name", "param_value", "trial_id", "timestep"])

    for (param_name, param_value, trial_id, timestep), group in grouped:
        reasons = group[reason_column].dropna().tolist()
        reasons = [r for r in reasons if r and len(r.strip()) > 0]

        if len(reasons) < 2:
            results.append({
                "param_name": param_name,
                "param_value": param_value,
                "trial_id": trial_id,
                "timestep": timestep,
                "semantic_variance": 0.0,
                "semantic_spread": 0.0,
                "polarization_index": 0.0,
                "mean_similarity": 1.0,
                "num_reasons": len(reasons)
            })
            continue

        embeddings = analyzer.compute_embeddings(reasons)

        results.append({
            "param_name": param_name,
            "param_value": param_value,
            "trial_id": trial_id,
            "timestep": timestep,
            "semantic_variance": analyzer.semantic_variance(embeddings),
            "semantic_spread": analyzer.semantic_spread(embeddings),
            "polarization_index": analyzer.polarization_index(embeddings),
            "mean_similarity": analyzer.mean_pairwise_similarity(embeddings),
            "num_reasons": len(reasons)
        })

    return pd.DataFrame(results)


def compute_semantic_metrics_by_personality(
    df: pd.DataFrame,
    analyzer: Optional[SemanticAnalyzer] = None,
    reason_column: str = "reason",
    personality_column: str = "personality"
) -> pd.DataFrame:
    if analyzer is None:
        analyzer = SemanticAnalyzer()

    results = []

    grouped = df.groupby(["param_name", "param_value", "trial_id", "timestep", personality_column])

    for (param_name, param_value, trial_id, timestep, personality), group in grouped:
        reasons = group[reason_column].dropna().tolist()
        reasons = [r for r in reasons if r and len(r.strip()) > 0]

        if len(reasons) < 2:
            continue

        embeddings = analyzer.compute_embeddings(reasons)

        results.append({
            "param_name": param_name,
            "param_value": param_value,
            "trial_id": trial_id,
            "timestep": timestep,
            "personality": personality,
            "semantic_variance": analyzer.semantic_variance(embeddings),
            "polarization_index": analyzer.polarization_index(embeddings),
            "mean_similarity": analyzer.mean_pairwise_similarity(embeddings),
            "num_reasons": len(reasons)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    if SBERT_AVAILABLE:
        analyzer = SemanticAnalyzer()

        test_reasons = [
            "AI regulation is necessary to prevent misuse and protect citizens",
            "Regulation will stifle innovation and hurt economic growth",
            "We need balanced oversight that doesn't impede progress",
            "Companies should be free to develop AI without government interference",
            "Strong regulation is the only way to ensure AI safety"
        ]

        embeddings = analyzer.compute_embeddings(test_reasons)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Semantic variance: {analyzer.semantic_variance(embeddings):.4f}")
        print(f"Semantic spread: {analyzer.semantic_spread(embeddings):.4f}")
        print(f"Polarization index: {analyzer.polarization_index(embeddings):.4f}")
        print(f"Mean pairwise similarity: {analyzer.mean_pairwise_similarity(embeddings):.4f}")

        print("\nPairwise similarity matrix:")
        sim_matrix = analyzer.pairwise_cosine_similarity(embeddings)
        print(np.round(sim_matrix, 3))
    else:
        print("SBERT not available. Install with: pip install sentence-transformers")
