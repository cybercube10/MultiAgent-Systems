import os
import re
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from readability import Document
from tavily import TavilyClient
import trafilatura
from rich import print
from langchain.tools import tool


load_dotenv()

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def web_search(query: str) -> str:
    """Search the web using Tavily and return concise results."""

    results = tavily.search(
        query=query,
        max_results=4
    )

    out = []

    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['content'][:300]}\n"
        )

    return "\n---\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Scrape and extract clean readable content from a URL.
    Uses multiple extraction strategies for better readability.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        html = response.text

        # Strategy 1: Trafilatura
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        )

        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r"\s+", " ", extracted).strip()
            return cleaned[:5000]

        # Strategy 2: Readability + BeautifulSoup
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(
            clean_html,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        if text and len(text.strip()) > 200:
            cleaned = re.sub(r"\s+", " ", text).strip()
            return cleaned[:5000]

        return "Could not extract meaningful content from the URL."

    except requests.exceptions.Timeout:
        return "Request timed out while scraping the URL."

    except requests.exceptions.HTTPError as e:
        return f"HTTP error occurred: {e}"

    except requests.exceptions.RequestException as e:
        return f"Request failed while scraping the URL: {e}"

    except Exception as e:
        return f"Could not scrape URL: {e}"

