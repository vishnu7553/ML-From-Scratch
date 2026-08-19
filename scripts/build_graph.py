#!/usr/bin/env python3
"""Extract the Obsidian wikilink graph from Notes/ and emit docs/graph-data.js.

Nodes: every note in Notes/ (concept) plus every implementation file it links
to. Edges: every [[...]] link, with [[A|Alias]] resolved to A.

Links are resolved case-insensitively against the actual filenames, so a
wikilink like [[Optimal Margin Classifier]] still connects to the real note
"Optimal margin classifier.md" instead of silently disappearing.

Run from the repo root:
    python scripts/build_graph.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "Notes"
OUT = ROOT / "docs" / "graph-data.js"

LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
ALIAS_RE = re.compile(r"^(.*?)\|.*$")

CATEGORIES = {
    "Foundations": {
        "Machine Learning",
        "Supervised Learning",
    },
    "Regression": {
        "Linear Regression",
        "Locally weighted regression",
        "Probabilistic Interpretation",
        "Overfitting and Underfitting",
    },
    "Classification": {
        "Logistic Regression",
        "Newton's method",
        "Exponential family",
        "Generalized linear models",
        "Softmax Regression",
        "Gradient Descent",
        "Loss Functions",
        "Sigmoid Function",
    },
    "Generative": {
        "Generative Learning Algorithms",
        "Gaussian Discriminant Analysis",
        "Naive bayes",
    },
    "SVM": {
        "Support Vector Machines",
        "Optimal margin classifier",
        "Kernels",
    },
}


def category_for(title):
    for cat, titles in CATEGORIES.items():
        if title in titles:
            return cat
    return "Other"


def note_title_from_path(path):
    """Resolve a relative link target to a note title or implementation path."""
    parts = path.split("/")
    if "Implementation" in parts:
        rel = Path(*parts[parts.index("Implementation") + 1:])
        return str(rel)
    return path.replace("../", "").replace(".md", "")


def build_graph():
    notes = {}
    by_lower = {}
    for f in sorted(NOTES_DIR.glob("*.md")):
        notes[f.stem] = f
        by_lower[f.stem.lower()] = f.stem

    nodes = []
    edges = []
    seen_nodes = set()

    def add_node(title, kind):
        key = title if kind == "note" else f"impl:{title}"
        if key in seen_nodes:
            return key
        seen_nodes.add(key)
        nodes.append({
            "id": key,
            "label": title,
            "kind": kind,
            "category": category_for(title) if kind == "note" else "Implementation",
            "file": f"Notes/{title}.md" if kind == "note" else f"Implementation/{title}",
        })
        return key

    def resolve(target):
        """Resolve a wikilink to an existing note title, case-insensitively."""
        if target.endswith(".py"):
            return target
        title = target.replace("../", "").replace(".md", "")
        if title in notes:
            return title
        return by_lower.get(title.lower())

    for title in sorted(notes):
        src = add_node(title, "note")
        text = notes[title].read_text(encoding="utf-8")
        for raw in LINK_RE.findall(text):
            target = ALIAS_RE.sub(r"\1", raw)
            resolved = resolve(target)
            if resolved is None:
                continue
            if resolved.endswith(".py"):
                dst = add_node(resolved, "impl")
                edges.append({"source": src, "target": dst, "type": "note->impl"})
            else:
                dst = add_node(resolved, "note")
                if src != dst:
                    edges.append({"source": src, "target": dst, "type": "note->note"})

    return nodes, edges


def main():
    nodes, edges = build_graph()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "window.GRAPH = " + json.dumps({"nodes": nodes, "edges": edges}, indent=2) + ";\n",
        encoding="utf-8",
    )
    n_notes = sum(1 for n in nodes if n["kind"] == "note")
    n_impls = sum(1 for n in nodes if n["kind"] == "impl")
    print(f"Wrote {len(nodes)} nodes ({n_notes} notes, {n_impls} implementations) "
          f"and {len(edges)} edges to {OUT}")


if __name__ == "__main__":
    sys.exit(main())