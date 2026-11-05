from app.domain import SearchResult


def build_prompt(query: str, sources: list[SearchResult]) -> str:
    evidence = "\n\n".join(
        (
            f"[S{index}] {result.chunk.document_name} "
            f"({result.chunk.source_location or 'location unavailable'}"
            f"{', ' + result.chunk.section_heading if result.chunk.section_heading else ''})\n"
            f"{result.chunk.content}"
        )
        for index, result in enumerate(sources, start=1)
    )
    if not evidence:
        evidence = "No evidence was retrieved."
    return (
        "You are an assistant answering engineering questions. Use only the evidence "
        "provided below. Preserve numerical values, units, signs, symbols, and decimal "
        "precision exactly as written. Do not infer missing specifications. Cite "
        "supporting evidence using its [S#] label. If the evidence does not answer the "
        "question, state that the available documents are insufficient.\n\n"
        f"Evidence:\n{evidence}\n\nQuestion: {query}\n\nAnswer:"
    )
