"""Ingestion: turn the catalog and policy documents into chunks and embed them."""
import json
import re

import numpy as np

from .config import ROOT, load_jsonl

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_DIR = ROOT / "index"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def product_text(p: dict) -> str:
    price = f"Price: ${p['price']:.2f}."
    if p.get("sale_price"):
        price += f" On sale for ${p['sale_price']:.2f} (sale item)."
    status = ("Status: discontinued, no longer available." if p["status"] == "discontinued"
              else "Status: in the current range.")
    return (f"{p['name']} (SKU {p['sku']}). Category: {p['category']}. {price} {status} "
            f"Sizes: {p['sizes']}. Features: {'; '.join(p['features'])}.")


def build_chunks() -> list[dict]:
    """One chunk per product; one chunk per '## ' section of each policy file."""
    chunks = [{"id": p["sku"], "type": "product", "text": product_text(p)}
              for p in load_jsonl(ROOT / "data" / "catalog.jsonl")]
    for path in sorted((ROOT / "data" / "policies").glob("*.md")):
        for section in re.split(r"\n(?=## )", path.read_text(encoding="utf-8")):
            section = section.strip()
            if not section.startswith("## "):
                continue  # skip the file title
            heading, _, body = section[3:].partition("\n")
            chunks.append({"id": f"{path.stem}#{slug(heading)}", "type": "policy",
                           "text": f"{path.stem.replace('_', ' ').title()} policy - {heading.strip()}: {body.strip()}"})
    return chunks


def build_index() -> int:
    from sentence_transformers import SentenceTransformer

    chunks = build_chunks()
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors.astype("float32"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=1), encoding="utf-8")
    return len(chunks)
