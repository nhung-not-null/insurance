import random
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter


def load_markdown_chunks(path: str) -> list[str]:
    """Charge les fichiers markdown et les decoupe par titres de niveau 1.
    Accepte un chemin vers un fichier .md ou un dossier contenant des fichiers .md.
    """
    input_path = Path(path)
    file_chunks = []
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1")])
    md_files = [input_path] if input_path.is_file() else list(input_path.glob("*.md"))
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            sections = splitter.split_text(f.read())
            for section in sections:
                header = section.metadata.get("Header 1", "Contrat")
                file_chunks.append(f"Source: {md_file.name} | Section: {header}\n\n{section.page_content}")
    return file_chunks


def load_markdown_chunks_small(path: str) -> list[str]:
    """Decoupe les fichiers par titres de niveau 2 et 3 (## et ###) pour creer de petits chunks.
    Accepte un chemin vers un fichier .md ou un dossier contenant des fichiers .md.
    """
    input_path = Path(path)
    small_chunks = []
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Header 2"), ("###", "Header 3")])
    md_files = [input_path] if input_path.is_file() else list(input_path.glob("*.md"))
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            sections = splitter.split_text(f.read())
            for section in sections:
                h2 = section.metadata.get("Header 2", "")
                h3 = section.metadata.get("Header 3", "")
                header = " > ".join(filter(None, [h2, h3])) or "Contrat"
                content = section.page_content.strip()
                if content:
                    small_chunks.append(f"Source: {md_file.name} | Section: {header}\n\n{content}")
    return small_chunks


def create_chunk_combinations(
    small_chunks: list[str],
    num_combinations: int = 100,
    combo_sizes: tuple[int, ...] = (2, 3),
    seed: int = 42,
) -> list[str]:
    """Cree des combinaisons aleatoires de petits chunks pour enrichir le dataset.
    Echantillonne directement num_combinations combinaisons sans generer toutes les possibilites.

    Args:
        small_chunks: liste de petits chunks (H2/H3)
        num_combinations: nombre exact de combinaisons a produire
        combo_sizes: tuple des tailles de combinaisons a generer, ex: (2, 3)
        seed: graine pour la reproductibilite

    Returns:
        liste de chaines combinant plusieurs chunks
    """
    rng = random.Random(seed)
    n = len(small_chunks)
    seen: set[tuple[int, ...]] = set()
    combined_chunks: list[str] = []
    sizes_cycle = list(combo_sizes)
    max_attempts = num_combinations * 50
    attempts = 0
    while len(combined_chunks) < num_combinations and attempts < max_attempts:
        size = sizes_cycle[attempts % len(sizes_cycle)]
        attempts += 1
        if n < size:
            continue
        combo = tuple(sorted(rng.sample(range(n), size)))
        if combo in seen:
            continue
        seen.add(combo)
        combined_chunks.append("\n\n---\n\n".join(small_chunks[j] for j in combo))
    rng.shuffle(combined_chunks)
    print(f"Generated {len(combined_chunks)} combined chunks from {n} small chunks.")
    return combined_chunks
