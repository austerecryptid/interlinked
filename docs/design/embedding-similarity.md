# Type 4 Code Similarity — Local Embedding System

## Problem

LLM-generated codebases produce Type 4 duplicates constantly — structurally different
implementations of the same intent, written in separate conversations. Classical AST
comparison (Types 1-3) misses these because the code *looks* different even though it
*does* the same thing.

## Architecture

### Device Selection

```python
if torch.cuda.is_available():
    device = "cuda"        # ~2-5ms/function, 60k in 2-5 min
elif torch.backends.mps.is_available():
    device = "mps"         # Apple Silicon, similar to GPU
else:
    device = "cpu"         # ~20-50ms/function, 60k in 20-50 min (background)
```

### Model

UniXcoder (~130M params) or CodeBERT (~110M params). Small enough for CPU, fast on GPU.
Produces 768-dim vectors per function. Trained specifically on code similarity tasks.

### Persistence — `.interlinked/embeddings.db` (SQLite)

```sql
CREATE TABLE embeddings (
    function_id   TEXT PRIMARY KEY,
    source_hash   TEXT NOT NULL,      -- SHA256 of function source
    embedding     BLOB NOT NULL,      -- 768 float32 = 3KB per function
    model_version TEXT NOT NULL,
    created_at    REAL NOT NULL
);
```

- On startup: load existing embeddings, only recompute where `source_hash` changed
- First run: minutes (GPU) to tens of minutes (CPU)
- Subsequent runs: seconds (delta only)
- Model version change: full re-embed (rare)
- 60k functions × 3KB = ~180MB on disk

### Delta Updates (via existing file watcher)

1. File changes → parser rebuilds functions → new ASTs
2. Compare `source_hash` of each function against DB
3. Re-embed only changed functions (typically 5-20 per save)
4. Update SQLite + in-memory ANN index
5. Cost per save: < 100ms

### Progress / Availability

```python
class EmbeddingIndex:
    status: Literal["idle", "building", "ready"]
    progress: float    # 0.0 - 1.0
    total: int
    completed: int
    device: str        # "cuda", "mps", "cpu"
```

- Runs in background thread on startup
- Progress pushed via existing SSE channel → frontend progress bar
- `find_duplicates` / `similar_to` behavior by status:
  - `"ready"` → blend embedding cosine similarity into `_similarity_score`
  - `"building"` → classical-only, tool response includes "Embedding index building (43%)..."
  - `"idle"` → optional deps not installed, classical-only permanently
- MCP tool responses include status so LLM knows not to rely on embeddings until ready

### Integration with Existing Similarity

Embeddings don't replace classical scoring — they augment it:

```python
def _similarity_score(fp_a, fp_b, emb_a=None, emb_b=None):
    # ... existing classical signals (AST, Jaccard, control flow) ...
    
    # Embedding signal (when available)
    if emb_a is not None and emb_b is not None:
        emb_sim = cosine_similarity(emb_a, emb_b)
        scores.append((emb_sim, 4.0))  # Heaviest weight — captures intent
    
    return weighted_average(scores)
```

Classical features provide explainability ("differs by 3 AST edits").
Embedding captures intent ("both validate input then write to store").

### Optional Dependency

```toml
[project.optional-dependencies]
similarity = ["transformers", "torch"]
```

Core tool works without it. `pip install interlinked[similarity]` enables Type 4 detection.

## Pre-requisites (do first)

Before building this, upgrade the classical kernel:
1. **Minhash pre-filter** — O(n²) → near-linear for pairwise comparison
2. **Tree Edit Distance** on ASTs — replaces cosine-on-counts with structural comparison  
3. **WL graph kernel** on local call neighborhoods — uses NetworkX properly

These improvements are pure Python, zero new dependencies, and benefit both
the classical-only and hybrid paths.

## Estimated Implementation

- Classical upgrades (minhash, TED, WL): ~2-3 hours
- Embedding system (model, persistence, delta, progress): ~4-6 hours
- Frontend progress bar + status indicator: ~1 hour
- Testing on large codebase: ~2 hours
