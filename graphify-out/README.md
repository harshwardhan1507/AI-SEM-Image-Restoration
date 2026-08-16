# Graphify Knowledge Graph

This directory contains the Graphify knowledge graph and interactive visualization for the **AI SEM Image Restoration** codebase.

Graphify extracts codebase architecture, code symbols, dependencies, function calls, imports, and community clusters using deterministic Tree-Sitter AST parsing.

---

## 📁 Artifacts

- **[graph.html](file:///d:/Programming/python/semicon/graphify-out/graph.html)**: Interactive visual network graph viewer (Open in any modern browser).
- **[GRAPH_REPORT.md](file:///d:/Programming/python/semicon/GRAPH_REPORT.md)**: Comprehensive architectural report detailing community clusters and structural relationships.
- **[graph.json](file:///d:/Programming/python/semicon/graphify-out/graph.json)**: JSON graph database containing all extracted nodes (1,068 nodes) and edges (2,037 edges).
- **[manifest.json](file:///d:/Programming/python/semicon/graphify-out/manifest.json)**: Graph metadata and execution statistics.

---

## 🚀 How to Run & Update Graphify

### 1. Installation

Install `graphifyy` via `uv` or `pip`:

```bash
# Recommended via uv:
uv tool install graphifyy

# Or via Python pip:
python -m pip install graphifyy
```

### 2. Register Agent Skill (Gemini CLI)

To enable automatic AI context querying and post-edit updates:

```bash
graphify gemini install
```

This updates `GEMINI.md` and `.gemini/settings.json` with AST-only graph hooks.

---

## 🔍 Building & Updating the Graph

### Generate Knowledge Graph (Code AST Only)

To generate or update the graph without needing LLM API keys for non-code files:

```bash
graphify . --code-only
```

### Full Graph Generation (With LLM Summaries)

If `GEMINI_API_KEY` or `GOOGLE_API_KEY` is exported:

```bash
export GOOGLE_API_KEY="your-api-key"
graphify .
```

### Re-cluster & Update Architectural Report

To re-run community clustering and update `GRAPH_REPORT.md` and `graph.html`:

```bash
graphify cluster-only .
```

---

## 💡 Querying the Knowledge Graph

You can query relationships, modules, and concepts directly from your terminal:

### 1. Query a Question
```bash
graphify query "How does NAFBlock handle channel attention?"
```

### 2. Find Dependency Paths Between Components
```bash
graphify path "scripts/evaluate.py" "src/models/nafnet.py"
```

### 3. Explain a Specific Concept / Class
```bash
graphify explain "CheckpointManager"
```

---

## 🌐 Viewing the Visual Graph

Open `graphify-out/graph.html` directly in a browser to explore the interactive visual network:

- **Hover & Click**: Click any node to inspect imports, callers, and definitions in the sidebar.
- **Search**: Use the top-right search box to quickly highlight specific functions, classes, or files.
- **Filter**: Toggle community checkboxes to isolate specific modules (e.g., `NAFNet`, `Evaluator`, `CheckpointManager`).
