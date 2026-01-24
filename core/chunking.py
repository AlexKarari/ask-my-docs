"""
Chunking module.

Why chunking matters:
- The retriever can only retrieve chunks.
- If chunks split important ideas in half, retrieval becomes unreliable.
- Markdown-aware chunking preserves section boundaries, improving accuracy.

Strategy:
1) Split by Markdown headings (#, ##, ###)
2) If a section is still too large, sub-split with overlap.
"""

import re
from typing import List, Tuple

# Pattern explanation:
# - ^(#{1,6}): Matches 1-6 hash symbols at the start of a line (captured as group 1)
# - \s+: Matches one or more whitespace characters after the hashes
# - (.+)$: Matches the rest of the line as the heading text (captured as group 2)
# - re.MULTILINE: Makes ^ match at the start of each line, not just the document start
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Find sections in doc
def split_markdown_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split markdown text into (section_title, section_body) tuples.

    If the file has no headings, we return a single "Document" section.
    """
    # find all heading matches in the document
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("Document", text)]

    # Build list of sections by extracting text between consecutive headings
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Extract the heading components from the regex match groups
        heading_level = m.group(1)  # e.g., "##"
        heading_title = m.group(2).strip()
        section_text = text[start:end].strip()

        # Reconstruct the full heading string with level and title then store as tuples
        title = f"{heading_level} {heading_title}"
        sections.append((title, section_text))

    return sections

# Handle oversized text
def split_with_overlap(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Split a long text into overlapping chunks.

    Overlap helps when an answer spans the boundary between chunks.
    """
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + max_chars]
        chunks.append(chunk)
        if i + max_chars >= len(text):
            break
        i += max_chars - overlap
    return chunks

# combine function 1 & 2 above 
def chunk_markdown(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Markdown-aware chunking:
    - Split into sections by headings
    - Keep sections intact if they fit
    - Otherwise sub-split with overlap

    The two-tier strategy ensures:
    1. Small sections stay intact, preserving their semantic coherence
    2. Large sections are split responsibly with overlap to maintain context
    
    Args:
        text: The markdown document to chunk
        max_chars: Maximum characters per chunk (default: 1200, tuned for embedding models)
        overlap: Characters to overlap between sub-chunks (default: 200)

    Returns:
        A list of chunk strings, each respecting markdown structure and size constraints
    """
    chunks = []
    sections = split_markdown_into_sections(text)

    for title, body in sections:
        # Keep the title with the section body to preserve context.
        section_text = f"{title}\n{body}".strip()

        if len(section_text) <= max_chars:
            chunks.append(section_text)
        else:
            # If too long, split with overlap
            chunks.extend(split_with_overlap(section_text, max_chars, overlap))

    return chunks
