import arxiv
from typing import Dict, Any, List
from vinagent.register import primary_function


@primary_function
def paper_research(topic: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for academic papers on arXiv and return paper information.

    Args:
        topic (str): Research topic or keywords to search for.
        max_results (int): Maximum number of papers to retrieve.

    Returns:
        Dict[str, Any]: Dictionary containing paper IDs and detailed paper information.
    """

    client = arxiv.Client()

    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    paper_ids = []
    papers_info = []

    for paper in papers:
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)

        paper_info = {
            "paper_id": paper_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
        }

        papers_info.append(paper_info)

    result = {
        "paper_ids": paper_ids,
        "papers_info": papers_info,
        "total_papers": len(paper_ids),
        "topic": topic,
    }

    return result
