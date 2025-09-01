import arxiv
import json
import os
import tempfile
from typing import List
from vinagent.register import primary_function


PAPER_DIR = os.path.join(tempfile.gettempdir(), "vinagent_papers")


@primary_function
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for academic papers on arXiv based on a research topic and store their metadata locally.

    This function queries the arXiv database for papers matching the given topic, retrieves
    detailed information about each paper, and stores the data in a structured JSON format
    for later retrieval. The papers are sorted by relevance to the search topic.

    Args:
        topic (str): The research topic or keywords to search for. Can include specific
                    terms, author names, or broad subject areas (e.g., "machine learning",
                    "neural networks", "transformer architecture")
        max_results (int, optional): Maximum number of papers to retrieve and store.
                                   Defaults to 5. Recommended range: 1-20 for optimal performance.

    Returns:
        List[str]: A list of arXiv paper IDs (short format, e.g., ['2301.07041', '2012.11747'])
                  that can be used with extract_paper_info() to get detailed information.

    Example:
        >>> paper_ids = search_papers("attention mechanism", max_results=3)
        >>> print(paper_ids)
        ['2301.07041', '2012.11747', '1706.03762']

    Note:
        - Paper metadata is automatically saved to a JSON file in the system temp directory
        - Creates topic-specific subdirectories for organized storage
        - Existing data is preserved and new papers are added to the collection
        - Internet connection required for arXiv API access
    """

    # Use arxiv to find the papers
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
        }
        papers_info[paper.get_short_id()] = paper_info

    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")

    return paper_ids
