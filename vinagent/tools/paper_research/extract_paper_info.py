import arxiv
import json
import os
import tempfile
from typing import List
from vinagent.register import primary_function

PAPER_DIR = os.path.join(tempfile.gettempdir(), "vinagent_papers")


@primary_function
def extract_paper_info(paper_id: str) -> str:
    """
    Retrieve detailed information about a specific academic paper using its arXiv ID.

    This function searches through all locally stored paper databases to find information
    about the specified paper. It returns comprehensive metadata including title, authors,
    abstract, publication date, and PDF URL if the paper was previously downloaded via
    search_papers().

    Args:
        paper_id (str): The arXiv paper identifier in short format (e.g., '2301.07041',
                       '1706.03762'). This should be an ID returned from search_papers()
                       or a known arXiv paper ID.

    Returns:
        str: JSON-formatted string containing paper details if found, including:
             - title: Full title of the paper
             - authors: List of author names
             - summary: Paper abstract/summary
             - pdf_url: Direct link to the PDF
             - published: Publication date (YYYY-MM-DD format)

             If paper not found: Error message indicating no saved information exists.

    Example:
        >>> info = extract_paper_info("1706.03762")
        >>> print(info)
        {
          "title": "Attention Is All You Need",
          "authors": ["Ashish Vaswani", "Noam Shazeer", ...],
          "summary": "The dominant sequence transduction models...",
          "pdf_url": "http://arxiv.org/pdf/1706.03762v5.pdf",
          "published": "2017-06-12"
        }

    Note:
        - Only returns information for papers previously searched with search_papers()
        - Searches across all topic directories in the local storage
        - Returns raw JSON string that may need parsing for programmatic use
        - No internet connection required (uses local cache)
    """

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."
