# Finding Trending new on Google New

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/docs/tutorials/guides/2.Trending_New.ipynb)

Keeping up with trending information about a specific company is crucial for investment decisions. By collecting a set of key news items in a timely manner, you can proactively mitigate risks and seize lucrative opportunities. This tutorial will guide you through designing a Trending Search Agent to collect news efficiently and on time.


## Install libraries


```python
%pip install vinagent 
%pip install tavily-python=0.3.1 googlenewsdecoder=0.1.7 langchain-together=0.3.0
```

## Setup environment variables

To use a list of default tools inside [vinagent.tools](vinagent/tools/) you should set environment varibles inside `.env` including `TOGETHER_API_KEY` to use llm models at [togetherai](https://api.together.ai/signin) site and `TAVILY_API_KEY` to use tavily websearch tool at [tavily](https://app.tavily.com/home) site:


```python
%%writefile .env
TOGETHER_API_KEY=your_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Design trending tools

We leverage Google News to search for a list of RSS links related to a particular topic, and then use a decoding method to parse the content of each article. An LLM is used to summarize the key points of each article and organize them into a list of trending articles.


```python
%%writefile vinagent/tools/trending_news.py
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
```

## Initialize your LLM and Agent


```python
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env')) # Replace by your own .env absolute path file

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

agent = Agent(
    description="You are a Financial Analyst",
    llm = llm,
    skills = [
        "Deeply analyzing financial markets", 
        "Searching information about stock price",
        "Visualization about stock price"],
    tools = [
        'vinagent.tools.trending_news'
    ],
    tools_path = 'templates/tools.json',
    is_reset_tools = True
)
```

    INFO:httpx:HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.register.tool:Registered trending_news_google_tools:
    {'tool_name': 'trending_news_google_tools', 'arguments': {'top_k': 5, 'topic': 'AI', 'host_language': 'en-US', 'geo_location': 'US'}, 'return': 'a list of dictionaries containing the title, link, and summary of the top trending news', 'docstring': 'Summarize the top trending news from Google News from a given topic.', 'dependencies': ['logging', 're', 'typing', 'requests', 'dotenv', 'pandas', 'bs4', 'urllib.parse', 'langchain_together', 'googlenewsdecoder'], 'module_path': 'vinagent.tools.trending_news', 'tool_type': 'module', 'tool_call_id': 'tool_64ac41d7-450e-4ca1-8280-9fd3c37dc40c'}
    INFO:vinagent.register.tool:Registered TrendingTopics.get_ai_news:
    {'tool_name': 'TrendingTopics.get_ai_news', 'arguments': {'top_k': 5, 'topic': 'artificial intelligence', 'host_language': 'en-US', 'geo_location': 'US'}, 'return': 'pd.DataFrame: DataFrame containing article links', 'docstring': 'Fetch top 10 AI news articles.', 'dependencies': ['logging', 're', 'typing', 'requests', 'dotenv', 'pandas', 'bs4', 'urllib.parse', 'langchain_together', 'googlenewsdecoder'], 'module_path': 'vinagent.tools.trending_news', 'tool_type': 'module', 'tool_call_id': 'tool_c0f25283-ee65-4381-a91c-63d4c62a3466'}
    INFO:vinagent.register.tool:Registered TrendingTopics.get_summary:
    {'tool_name': 'TrendingTopics.get_summary', 'arguments': {'news_id': 0}, 'return': 'Dict', 'docstring': 'Generate JSON summary for a news article.', 'dependencies': ['logging', 're', 'typing', 'requests', 'dotenv', 'pandas', 'bs4', 'urllib.parse', 'langchain_together', 'googlenewsdecoder'], 'module_path': 'vinagent.tools.trending_news', 'tool_type': 'module', 'tool_call_id': 'tool_3b64284b-e858-43f4-9fec-fc9c7d85de50'}
    INFO:vinagent.register.tool:Registered TrendingTopics.summarize_article:
    {'tool_name': 'TrendingTopics.summarize_article', 'arguments': {'title': '', 'source_url': ''}, 'return': 'Optional[str]', 'docstring': 'Generate structured article summary.', 'dependencies': ['logging', 're', 'typing', 'requests', 'dotenv', 'pandas', 'bs4', 'urllib.parse', 'langchain_together', 'googlenewsdecoder'], 'module_path': 'vinagent.tools.trending_news', 'tool_type': 'module', 'tool_call_id': 'tool_647b02a0-66ac-4b49-9764-99d42ab41f61'}
    INFO:vinagent.register.tool:Registered TrendingTopics.extract_text_from_rss_url:
    {'tool_name': 'TrendingTopics.extract_text_from_rss_url', 'arguments': {'rss_url': ''}, 'return': 'Optional[str]', 'docstring': 'Extract cleaned text from RSS URL.', 'dependencies': ['logging', 're', 'typing', 'requests', 'dotenv', 'pandas', 'bs4', 'urllib.parse', 'langchain_together', 'googlenewsdecoder'], 'module_path': 'vinagent.tools.trending_news', 'tool_type': 'module', 'tool_call_id': 'tool_c9369568-fbaa-4a7e-a3a0-739efae35cfb'}
    INFO:vinagent.register.tool:Registered TrendingTopics.decode_rss_url:
    {'tool_name': 'TrendingTopics.decode_rss_url', 'arguments': {'source_url': ''}, 'return': 'Optional[str]', 'docstring': 'Decode Google News RSS URL.', 'dependencies': ['logging', 're', 'typing', 'requests', 'dotenv', 'pandas', 'bs4', 'urllib.parse', 'langchain_together', 'googlenewsdecoder'], 'module_path': 'vinagent.tools.trending_news', 'tool_type': 'module', 'tool_call_id': 'tool_ec0cb8c7-743c-4a8c-b753-4ef0a969c4f6'}
    INFO:vinagent.register.tool:Registered TrendingTopics._is_valid_url:
    {'tool_name': 'TrendingTopics._is_valid_url', 'arguments': {'url': ''}, 'return': 'bool', 'docstring': 'Validate URL format.', 'dependencies': ['logging', 're', 'typing', 'requests', 'dotenv', 'pandas', 'bs4', 'urllib.parse', 'langchain_together', 'googlenewsdecoder'], 'module_path': 'vinagent.tools.trending_news', 'tool_type': 'module', 'tool_call_id': 'tool_fdb1acc0-0086-4ca9-af29-c122100c854a'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.trending_news


## Asking your agent


```python
message = agent.invoke("""Let's find the top 5 trending news about NVIDIA today.""")
```


```python
from IPython.display import Markdown, display
display(Markdown(message.artifact))
```


    # Where Will Nvidia Stock Be in 10 Years? - Yahoo Finance
    ## What is new?
    Nvidia's generative AI business is still performing well, but there are signs of slowing growth. The company's revenue growth has decelerated to 69% from 262% in the previous fiscal quarter. Additionally, new technologies like self-driving cars and robotics could be key to Nvidia's long-term success, with potential annual revenue of $300 billion to $400 billion by 2035 for self-driving technology and $38 billion for humanoid robots.

    ## Highlight
    The key points of the article include: Nvidia's data center business represents 89% of its total revenue, the company's AI chip business may be slowing down, and new business verticals like robotics and self-driving cars could help diversify Nvidia's revenue streams. The company's automation and robotics segment has already shown significant growth, with first-quarter sales jumping 72% year over year to $567 million.

    ## Why it matters
    The potential slowing down of Nvidia's AI chip business and the company's ability to pivot to new technologies will have a significant impact on its long-term success. If Nvidia can successfully transition to new business verticals, it could maintain its dominant position in the market and continue to thrive. However, if it fails to adapt to changing conditions, it may experience stagnation or decline, as has been the case with other companies that have failed to evolve with technological advancements.

    ## Link
    https://finance.yahoo.com/news/where-nvidia-stock-10-years-200000792.html

    # Nvidia's latest DLSS revision reduces VRAM usage by 20% for upscaling — optimizations reduce overhead of more powerful transformer model - Tom's Hardware
    ## What is new?
    Nvidia has released a new revision of its DLSS (Deep Learning Super Sampling) technology, which reduces VRAM usage by 20% for upscaling. This update optimizes the transformer model, making it more efficient and reducing its memory footprint. The new revision, DLSS 310.3.0, improves the transformer model's VRAM usage, bringing it closer to the older CNN model's memory impact.

    ## Highlight
    The key points of this update include:
    * 20% reduction in VRAM usage for upscaling
    * Optimizations reduce the overhead of the more powerful transformer model
    * The new transformer model consumes 40% more memory than the CNN model, down from nearly twice as much
    * Memory consumption increases linearly with resolution, with the transformer model consuming 85.77MB of VRAM at 1080p and 307.37MB at 4K

    ## Why it matters
    This update is significant because it shows Nvidia's commitment to improving the efficiency of its DLSS technology. While the 20% reduction in VRAM usage may not have a noticeable impact on real-world applications, it demonstrates the company's efforts to optimize its technology for better performance. Additionally, the reduction in memory footprint could be beneficial for systems with limited VRAM, particularly at higher resolutions like 8K. This update also highlights the ongoing development and refinement of DLSS, which is now used in over 760 games and apps.

    ## Link
    https://www.tomshardware.com/pc-components/gpus/nvidias-latest-dlss-revision-reduces-vram-usage-by-20-percent-for-upscaling-optimizations-reduce-overhead-of-more-powerful-transformer-model

    # Nvidia executives cash out $1bn worth of shares - Financial Times
    ## What is new?
    Nvidia executives have recently sold a substantial amount of shares, totaling $1 billion in value. This significant transaction has drawn attention to the company's internal dynamics and potential future directions.

    ## Highlight
    The key points of this news include the large-scale sale of Nvidia shares by its executives, amounting to $1 billion. This move could indicate a shift in the executives' confidence in the company's future prospects or a strategic decision to diversify their personal investments.

    ## Why it matters
    The sale of such a large volume of shares by Nvidia executives could have implications for investor confidence and the company's stock price. It may also signal potential changes in Nvidia's leadership or strategy, as significant insider transactions often attract scrutiny from investors and market analysts. Understanding the motivations behind this sale can provide insights into the company's future growth prospects and industry trends.

    ## Link
    https://www.ft.com/content/36f346ad-c649-42ac-a6b6-1a8cc881e0bb

    # Nvidia: The Music Is About To Stop (NASDAQ:NVDA) - Seeking Alpha
    ## What is new?
    The article discusses the potential risks and challenges facing Nvidia Corporation, including macro and geopolitical risks, rising competition, and their potential impact on the company's performance. The authors, Bears of Wall Street, maintain a bearish stance on NVDA stock, citing these factors as reasons to sell.

    ## Highlight
    The key points of the article include:
    * Nvidia's stock has risen around 15% since the last coverage before its Q1 earnings report
    * Macro and geopolitical risks could have a significant impact on Nvidia's performance
    * Rising competition may lead to lower demand for Nvidia's products in the future
    * The authors recommend a "Sell" position on NVDA stock due to these and other factors

    ## Why it matters
    The article's analysis matters because it highlights the potential risks and challenges that Nvidia faces, which could impact the company's future growth and profitability. Investors who are considering buying or holding NVDA stock should be aware of these risks and consider the authors' bearish stance when making their investment decisions. Additionally, the article's focus on macro and geopolitical risks, as well as rising competition, underscores the importance of considering broader market trends and industry dynamics when evaluating individual stocks.

    ## Link
    https://seekingalpha.com/article/4797785-nvidia-the-music-is-about-to-stop

