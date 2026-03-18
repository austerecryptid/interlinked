"""Code embedding index — local model for Type 4 semantic similarity detection.

Uses UniXcoder (or CodeBERT fallback) to produce dense vector embeddings of
function/method source code. Embeddings are persisted in SQLite and updated
incrementally via source hash comparison.

Device priority: CUDA GPU → Apple MPS → CPU.

Optional dependency: requires `transformers` and `torch`.
Install via: pip install interlinked[similarity]
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

# ── Availability check ────────────────────────────────────────────────

_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import transformers  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if embedding dependencies are installed."""
    return _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE


def _select_device() -> str:
    """Select best available compute device."""
    if not _TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Vector serialization ─────────────────────────────────────────────

def _pack_vector(vec: list[float]) -> bytes:
    """Pack a float vector to bytes for SQLite storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(data: bytes) -> list[float]:
    """Unpack bytes back to a float vector."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


# ── Source hashing ────────────────────────────────────────────────────

def _source_hash(source: str) -> str:
    """SHA256 hash of source text for change detection."""
    return hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()[:16]


# ── SQLite persistence ───────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    function_id   TEXT PRIMARY KEY,
    source_hash   TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    model_name    TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_hash ON embeddings(source_hash);
"""


class _EmbeddingDB:
    """Thread-safe SQLite wrapper for embedding persistence."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._local = threading.local()
        # Initialize schema on the main thread
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._path, check_same_thread=False)
        return self._local.conn

    def get(self, function_id: str) -> tuple[str, list[float]] | None:
        """Get (source_hash, embedding) for a function, or None."""
        row = self._conn().execute(
            "SELECT source_hash, embedding FROM embeddings WHERE function_id = ?",
            (function_id,),
        ).fetchone()
        if row:
            return row[0], _unpack_vector(row[1])
        return None

    def get_all(self) -> dict[str, tuple[str, list[float]]]:
        """Load all embeddings: {function_id: (source_hash, vector)}."""
        result = {}
        for row in self._conn().execute("SELECT function_id, source_hash, embedding FROM embeddings"):
            result[row[0]] = (row[1], _unpack_vector(row[2]))
        return result

    def upsert(self, function_id: str, source_hash: str, embedding: list[float], model_name: str) -> None:
        """Insert or update an embedding."""
        self._conn().execute(
            "INSERT OR REPLACE INTO embeddings (function_id, source_hash, embedding, model_name, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (function_id, source_hash, _pack_vector(embedding), model_name, time.time()),
        )
        self._conn().commit()

    def upsert_batch(self, rows: list[tuple[str, str, list[float], str]]) -> None:
        """Batch upsert: [(function_id, source_hash, embedding, model_name), ...]."""
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (function_id, source_hash, embedding, model_name, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(fid, sh, _pack_vector(emb), mn, time.time()) for fid, sh, emb, mn in rows],
        )
        conn.commit()

    def delete(self, function_ids: list[str]) -> None:
        """Remove embeddings for deleted functions."""
        if not function_ids:
            return
        conn = self._conn()
        placeholders = ",".join("?" * len(function_ids))
        conn.execute(f"DELETE FROM embeddings WHERE function_id IN ({placeholders})", function_ids)
        conn.commit()

    def clear_all(self) -> None:
        """Wipe the entire table (e.g. model version change)."""
        self._conn().execute("DELETE FROM embeddings")
        self._conn().commit()

    def count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]


# ── Model wrapper ────────────────────────────────────────────────────

# Model preference order — try UniXcoder first, then CodeBERT
_MODEL_CANDIDATES = [
    "microsoft/unixcoder-base",
    "microsoft/codebert-base",
]


class _CodeEmbedder:
    """Wraps a HuggingFace code model for embedding extraction."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name: str = ""
        self.device: str = "cpu"
        self._dim: int = 768

    def load(self) -> bool:
        """Load model onto best available device. Returns True on success."""
        if not is_available():
            logger.info("Embedding dependencies not installed (torch/transformers).")
            return False

        from transformers import AutoModel, AutoTokenizer

        self.device = _select_device()
        logger.info(f"Embedding device: {self.device}")

        for candidate in _MODEL_CANDIDATES:
            try:
                logger.info(f"Loading embedding model: {candidate}")
                self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                self.model = AutoModel.from_pretrained(candidate)
                self.model.to(self.device)
                self.model.eval()
                self.model_name = candidate
                # Detect embedding dimension
                with torch.no_grad():
                    dummy = self.tokenizer("def f(): pass", return_tensors="pt", truncation=True, max_length=512)
                    dummy = {k: v.to(self.device) for k, v in dummy.items()}
                    out = self.model(**dummy)
                    self._dim = out.last_hidden_state.shape[-1]
                logger.info(f"Model loaded: {candidate} (dim={self._dim}, device={self.device})")
                return True
            except Exception as e:
                logger.warning(f"Failed to load {candidate}: {e}")
                continue

        logger.warning("No embedding model could be loaded.")
        return False

    @property
    def dim(self) -> int:
        return self._dim

    def embed_batch(self, sources: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of source code strings. Returns list of vectors."""
        if self.model is None or self.tokenizer is None:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**tokens)
                # Mean pooling over token embeddings (ignore padding)
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                # Normalize to unit vectors for cosine similarity
                normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.extend(normed.cpu().tolist())

        return all_embeddings

    def embed_one(self, source: str) -> list[float]:
        """Embed a single source string."""
        result = self.embed_batch([source], batch_size=1)
        return result[0] if result else []


# ── Main embedding index ─────────────────────────────────────────────

class EmbeddingIndex:
    """Manages the embedding lifecycle: build, persist, query, delta-update.

    Status lifecycle:
        idle     → dependencies not installed, classical-only forever
        loading  → model is being loaded
        building → initial embedding pass in progress
        ready    → all embeddings computed, queries use them
    """

    def __init__(self, project_path: str | Path) -> None:
        self.project_path = Path(project_path).resolve()
        self._db_path = self.project_path / ".interlinked" / "embeddings.db"
        self._db: _EmbeddingDB | None = None
        self._embedder = _CodeEmbedder()
        self._vectors: dict[str, list[float]] = {}  # In-memory cache: fid → vector

        # Progress tracking
        self.status: Literal["idle", "loading", "building", "ready"] = "idle"
        self.progress: float = 0.0
        self.total: int = 0
        self.completed: int = 0
        self.device: str = "cpu"
        self.model_name: str = ""

        # Callbacks
        self._on_progress: Callable[[dict], None] | None = None
        self._build_thread: threading.Thread | None = None

    def set_progress_callback(self, callback: Callable[[dict], None]) -> None:
        """Set a callback for progress updates: callback(status_dict)."""
        self._on_progress = callback

    def _notify_progress(self) -> None:
        if self._on_progress:
            self._on_progress(self.status_dict())

    def status_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "progress": round(self.progress, 3),
            "total": self.total,
            "completed": self.completed,
            "device": self.device,
            "model": self.model_name,
            "vector_count": len(self._vectors),
        }

    # ── Build (background) ────────────────────────────────────────

    def build_async(
        self,
        functions: list[dict[str, Any]],
    ) -> None:
        """Start building embeddings in a background thread.

        functions: list of dicts with keys:
            id: str, source: str (function source code)
        """
        if not is_available():
            self.status = "idle"
            logger.info("Embedding index: idle (dependencies not installed).")
            return

        if self._build_thread and self._build_thread.is_alive():
            logger.warning("Build already in progress.")
            return

        self._build_thread = threading.Thread(
            target=self._build_sync,
            args=(functions,),
            daemon=True,
            name="embedding-build",
        )
        self._build_thread.start()

    def _build_sync(self, functions: list[dict[str, Any]]) -> None:
        """Synchronous build — runs in background thread."""
        try:
            self.status = "building"
            self._notify_progress()

            # Open DB and load existing embeddings FIRST (cheap)
            self._db = _EmbeddingDB(self._db_path)
            existing = self._db.get_all()

            # Determine what needs embedding before loading the model
            to_embed: list[dict[str, Any]] = []
            cached_count = 0

            for func in functions:
                fid = func["id"]
                source = func["source"]
                if not source:
                    continue
                sh = _source_hash(source)

                # Check cache
                cached = existing.get(fid)
                if cached and cached[0] == sh:
                    # Source unchanged — use cached embedding
                    self._vectors[fid] = cached[1]
                    cached_count += 1
                else:
                    to_embed.append({"id": fid, "source": source, "hash": sh})

            self.total = len(to_embed) + cached_count
            self.completed = cached_count
            self.progress = self.completed / max(self.total, 1)
            self._notify_progress()

            # If everything is cached, skip model loading entirely
            if not to_embed:
                logger.info(
                    f"Embedding index: {cached_count} cached, 0 to compute — skipping model load"
                )
                self.status = "ready"
                self._notify_progress()
                return

            # Load model only when we actually need to embed something
            self.status = "loading"
            self._notify_progress()

            if not self._embedder.load():
                self.status = "idle"
                self._notify_progress()
                return

            self.device = self._embedder.device
            self.model_name = self._embedder.model_name

            # Check model version — if different from what's in DB, wipe and re-scan
            if existing:
                sample_id = next(iter(existing))
                row = self._db._conn().execute(
                    "SELECT model_name FROM embeddings WHERE function_id = ?",
                    (sample_id,),
                ).fetchone()
                if row and row[0] != self._embedder.model_name:
                    logger.info(f"Model changed ({row[0]} → {self._embedder.model_name}), wiping cache.")
                    self._db.clear_all()
                    # Re-scan: everything needs re-embedding now
                    self._vectors.clear()
                    to_embed = [
                        {"id": f["id"], "source": f["source"], "hash": _source_hash(f["source"])}
                        for f in functions if f.get("source")
                    ]
                    cached_count = 0
                    self.total = len(to_embed)
                    self.completed = 0

            self.status = "building"
            self._notify_progress()

            logger.info(
                f"Embedding index: {cached_count} cached, {len(to_embed)} to compute "
                f"({self.total} total, device={self.device})"
            )

            # Embed in batches
            batch_size = 32 if self.device != "cpu" else 8
            batch_rows: list[tuple[str, str, list[float], str]] = []

            for i in range(0, len(to_embed), batch_size):
                batch = to_embed[i:i + batch_size]
                sources = [f["source"] for f in batch]
                vectors = self._embedder.embed_batch(sources, batch_size=batch_size)

                for func_info, vec in zip(batch, vectors):
                    self._vectors[func_info["id"]] = vec
                    batch_rows.append((func_info["id"], func_info["hash"], vec, self._embedder.model_name))

                self.completed += len(batch)
                self.progress = self.completed / max(self.total, 1)

                # Persist batch
                if len(batch_rows) >= 100:
                    self._db.upsert_batch(batch_rows)
                    batch_rows = []

                self._notify_progress()

            # Flush remaining
            if batch_rows:
                self._db.upsert_batch(batch_rows)

            self.status = "ready"
            self._notify_progress()
            logger.info(f"Embedding index ready: {len(self._vectors)} vectors.")

        except Exception as e:
            logger.error(f"Embedding build failed: {e}", exc_info=True)
            self.status = "idle"
            self._notify_progress()

    # ── Delta update (for file watcher) ───────────────────────────

    def update_functions(self, functions: list[dict[str, Any]]) -> None:
        """Incrementally update embeddings for changed/new functions.

        Call this after the file watcher updates the graph.
        Fast: only re-embeds functions whose source hash changed.
        """
        if self.status not in ("ready", "building"):
            return
        if not self._db:
            return

        to_embed: list[dict[str, Any]] = []
        for func in functions:
            fid = func["id"]
            source = func["source"]
            if not source:
                continue
            sh = _source_hash(source)
            cached = self._db.get(fid)
            if cached and cached[0] == sh:
                continue  # Unchanged
            to_embed.append({"id": fid, "source": source, "hash": sh})

        if not to_embed:
            return

        # Lazy model load — needed when initial build found everything cached
        if self._embedder.model is None:
            if not self._embedder.load():
                logger.warning("Embedding delta: model failed to load, skipping.")
                return
            self.device = self._embedder.device
            self.model_name = self._embedder.model_name

        sources = [f["source"] for f in to_embed]
        vectors = self._embedder.embed_batch(sources)
        rows = []
        for func_info, vec in zip(to_embed, vectors):
            self._vectors[func_info["id"]] = vec
            rows.append((func_info["id"], func_info["hash"], vec, self._embedder.model_name))
        self._db.upsert_batch(rows)
        logger.info(f"Embedding delta: updated {len(rows)} vectors.")

    def remove_functions(self, function_ids: list[str]) -> None:
        """Remove embeddings for deleted functions."""
        if not self._db:
            return
        for fid in function_ids:
            self._vectors.pop(fid, None)
        self._db.delete(function_ids)

    # ── Query ─────────────────────────────────────────────────────

    def get_embedding(self, function_id: str) -> list[float] | None:
        """Get the embedding vector for a function, or None."""
        return self._vectors.get(function_id)

    def cosine_similarity(self, id_a: str, id_b: str) -> float | None:
        """Compute cosine similarity between two function embeddings."""
        va = self._vectors.get(id_a)
        vb = self._vectors.get(id_b)
        if va is None or vb is None:
            return None
        # Vectors are already L2-normalized, so dot product = cosine similarity
        return sum(a * b for a, b in zip(va, vb))

    def find_nearest(self, function_id: str, top_k: int = 20, threshold: float = 0.5) -> list[tuple[str, float]]:
        """Find the most similar functions by embedding cosine similarity.

        Returns [(function_id, similarity_score), ...] sorted by score descending.
        """
        vec = self._vectors.get(function_id)
        if vec is None:
            return []

        results: list[tuple[str, float]] = []
        for fid, fvec in self._vectors.items():
            if fid == function_id:
                continue
            score = sum(a * b for a, b in zip(vec, fvec))
            if score >= threshold:
                results.append((fid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"
