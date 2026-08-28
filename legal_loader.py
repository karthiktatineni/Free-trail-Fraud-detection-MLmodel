"""
LEGAL DOCUMENT LOADER — Single Source of Truth
================================================
Parses and serves canonical legal documents from the /legal directory with frontmatter metadata.
"""

import os
import re
from typing import List, Dict, Optional, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGAL_DIR = os.path.join(BASE_DIR, "legal")

CANONICAL_SLUGS = ["privacy", "terms", "cookies", "refund", "aup", "dpa", "disclaimer"]


def _parse_frontmatter_and_content(file_path: str) -> Dict[str, Any]:
    """Reads a markdown file with optional YAML-like frontmatter."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    meta = {}
    content = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            content = parts[2].strip()

            for line in fm_text.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    meta[key] = val

    return {
        "slug": meta.get("slug", os.path.splitext(os.path.basename(file_path))[0].lower()),
        "title": meta.get("title", meta.get("slug", "").title()),
        "version": meta.get("version", "1.0"),
        "effective_date": meta.get("effective_date", "2026-08-28"),
        "last_updated": meta.get("last_updated", "2026-08-28"),
        "category": meta.get("category", "General"),
        "summary": meta.get("summary", ""),
        "content": content
    }


def load_legal_documents(include_content: bool = True) -> List[Dict[str, Any]]:
    """Loads all legal documents from the legal/ directory in canonical order."""
    if not os.path.exists(LEGAL_DIR):
        return []

    docs_map = {}
    for fname in os.listdir(LEGAL_DIR):
        if fname.endswith(".md") and fname != "README.md":
            fpath = os.path.join(LEGAL_DIR, fname)
            doc_data = _parse_frontmatter_and_content(fpath)
            slug = doc_data["slug"]
            if not include_content:
                doc_data.pop("content", None)
            docs_map[slug] = doc_data

    # Return ordered list according to canonical sequence
    ordered = []
    for slug in CANONICAL_SLUGS:
        if slug in docs_map:
            ordered.append(docs_map.pop(slug))

    # Append any remaining docs
    for remaining in docs_map.values():
        ordered.append(remaining)

    return ordered


def get_legal_document(slug: str) -> Optional[Dict[str, Any]]:
    """Fetches a specific legal document by slug."""
    slug = slug.lower().strip()
    slug_aliases = {
        "privacy-policy": "privacy",
        "terms-of-service": "terms",
        "cookie-policy": "cookies",
        "refund-policy": "refund",
        "acceptable-use": "aup",
        "data-processing-agreement": "dpa",
        "ai-disclaimer": "disclaimer",
        "ml-disclaimer": "disclaimer"
    }
    normalized_slug = slug_aliases.get(slug, slug)

    docs = load_legal_documents(include_content=True)
    for doc in docs:
        if doc["slug"] == normalized_slug:
            return doc

    return None
