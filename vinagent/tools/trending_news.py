import logging
import re
from typing import Optional, Dict
import requests
from dotenv import load_dotenv
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain_together import ChatTogether
from googlenewsdecoder import gnewsdecoder
from vinagent.register import primary_function

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
model = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")


class TrendingTopics:
    def __init__(self):
        self._news_cache = None
        self._cache_timestamp = None
        self._cache_duration = 300  # Cache for 5 minutes
        self._max_text_length = 10000  # Max characters for model input
        self._header_agent = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
        }

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format.
        Args:
            url (str): input link for validating.
        Returns:
            bool: True if it was right link, else False.
        """
        pattern = re.compile(r"^https?://[^\s/$.?#].[^\s]*$")
        return bool(pattern.match(url))

    def decode_rss_url(self, source_url: str) -> Optional[str]:
        """Decode Google News RSS URL.
        Args:
            source_url (str): Google News RSS URL.
        Returns:
            str: Decoded URL or None if decoding fails.
        """
        if not self._is_valid_url(source_url):
            logger.error("Invalid URL format: %s", source_url)
            return None

        try:
            decoded_url = gnewsdecoder(source_url, interval=1)
            if decoded_url.get("status"):
                return decoded_url["decoded_url"]
            logger.warning("Decoding failed: %s", decoded_url["message"])
            return None
        except Exception as e:
            logger.error("Error decoding URL %s: %s", source_url, str(e))
            return None

    def extract_text_from_rss_url(self, rss_url: str) -> Optional[str]:
        """Extract cleaned text from RSS URL.
        Args:
            - rss_url (str): Google News RSS URL.
        Returns:
            str: Cleaned text from the RSS URL or None if extraction fails.
        """
        if not self._is_valid_url(rss_url):
            logger.error("Invalid RSS URL: %s", rss_url)
            return None

        decoded_url = self.decode_rss_url(rss_url)
        if not decoded_url:
            return None

        try:
            response = requests.get(decoded_url, headers=self._header_agent, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")

            for elem in soup.find_all(["script", "style", "nav", "footer"]):
                elem.decompose()

            text = soup.get_text(separator="\n", strip=True)
            return text[: self._max_text_length]
        except requests.RequestException as e:
            logger.error("Error fetching URL %s: %s", decoded_url, str(e))
            return None

    def summarize_article(self, title: str, source_url: str) -> Optional[str]:
        """Generate structured article summary."""
        if not title or not self._is_valid_url(source_url):
            logger.error("Invalid title or URL: %s, %s", title, source_url)
            return None
        decoded_url = self.decode_rss_url(source_url)
        text_content = self.extract_text_from_rss_url(source_url)
        if not text_content:
            logger.warning("No text content extracted for %s", decoded_url)
            return None

        try:
            prompt = (
                "You are a searching assistant who are in charge of collecting the trending news."
                "Let's summarize the following crawled content by natural language, Markdown format."
                f"- The crawled content**: {text_content[:self._max_text_length]}\n"
                "Let's organize output according to the following structure:\n"
                f"# {title}\n"
                "## What is new?"
                "- Summarize novel insights or findings.\n"
                "## Highlight"
                "- Highlight the key points with natural language.\n"
                "## Why it matters"
                "- Analyze significance and impact that are more specific and individual. Not repeat the same content with 'Hightlight' and 'What is new?' sections.\n"
                "## Link"
                f"{decoded_url}\n\n"
            )
            response = model.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error("Error summarizing article %s: %s", title, str(e))
            return None

    def get_ai_news(
        self,
        top_k: int = 5,
        topic: str = "artificial intelligence",
        host_language: str = "en-US",
        geo_location: str = "US",
    ) -> Optional[pd.DataFrame]:
        """Fetch top 10 AI news articles.
        Args:
            - top_k: Number of articles to fetch.
            - topic (str): Search topic. Default is "artificial intelligence",
            - host_language (str): Set language of the search results. Default is "en-US".
            - geo_location (str): Set location of the search results. Default is "US".
        Returns:
            pd.DataFrame: DataFrame containing article links
        """
        query = "+".join(topic.split())
        url = f"https://news.google.com/rss/search?q={query}&hl={host_language}&gl={geo_location}"
        try:
            response = requests.get(url, headers=self._header_agent, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "xml")

            items = soup.find_all("item")[:top_k]
            news_list = [
                {
                    "id": idx,
                    "title": item.title.text,
                    "link": item.link.text,
                    "published_date": item.pubDate.text,
                    "source": item.source.text if item.source else "Unknown",
                    "summary": "",
                }
                for idx, item in enumerate(items)
            ]
            self._news_cache = pd.DataFrame(news_list)
            self._cache_timestamp = pd.Timestamp.now()
            return self._news_cache
        except requests.RequestException as e:
            logger.error("Error fetching news: %s", str(e))
            return None

    def get_summary(self, news_id: int) -> Dict:
        """Generate JSON summary for a news article."""
        try:
            if not isinstance(news_id, int) or news_id < 0:
                return {"success": False, "error": "Invalid news ID"}

            if self._news_cache is None or self._news_cache.empty:
                return {"success": False, "error": "Failed to fetch news data"}

            if news_id >= len(self._news_cache):
                return {"success": False, "error": f"Invalid news ID: {news_id}"}

            article = self._news_cache.iloc[news_id]
            summary = self.summarize_article(article["title"], article["link"])

            if not summary:
                return {"success": False, "error": "Failed to generate summary"}

            return {"success": True, "summary": summary}
        except Exception as e:
            logger.error("Error in get_summary for ID %d: %s", news_id, str(e))
            return {"success": False, "error": f"Server error: {str(e)}"}


@primary_function
def trending_news_google_tools(
    top_k: int = 5,
    topic: str = "AI",
    host_language: str = "en-US",
    geo_location: str = "US",
) -> list[dict]:
    """
    Summarize the top trending news from Google News from a given topic.
    Args:
        - top_k: Number of articles to fetch.
        - topic (str): Search topic. Default is "artificial+intelligence",
        - host_language (str): Language of search results ('en-US', 'vi-VN', 'fr-FR'). Default is 'en-US'.
        - geo_location (str): Location of search results (e.g., 'US', 'VN', 'FR'). Default is 'US'.
    Returns:
        a list of dictionaries containing the title, link, and summary of the top trending news.
    """
    trending = TrendingTopics()
    news_df = trending.get_ai_news(
        top_k=top_k, topic=topic, host_language=host_language, geo_location=geo_location
    )
    news = []
    if news_df is not None:
        for i in range(len(news_df)):
            summary_i = trending.get_summary(i)
            logger.info(summary_i)
            news.append(summary_i)
    content = "\n\n".join([item["summary"] for item in news if "summary" in item])
    return content
