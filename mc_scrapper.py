import requests
from datetime import date
from bs4 import BeautifulSoup
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset

# Reuse TCP connections
session = requests.Session()

def get_soup(url: str) -> Optional[BeautifulSoup]:
    """Fetch and parse HTML from a given URL."""
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None

def scrape_article(url: str) -> Optional[dict]:
    """Scrape a Moneycontrol article by URL."""
    soup = get_soup(url)
    if not soup:
        return None

    title_tag = soup.find('h1', class_='article_title artTitle')
    content_wrapper = soup.find('div', class_='content_wrapper arti-flow', id='contentdata')
    if not title_tag:
        return None

    title = title_tag.get_text(strip=True)
    content = ''
    if content_wrapper:
        paragraphs = content_wrapper.find_all('p')
        content = '\n'.join(p.get_text(strip=True) for p in paragraphs)

    return {
        'title': title,
        'content': content,
        'date': date.today().isoformat(),
        'url': url
    }

def collect_article_links(url: str) -> Tuple[List[str], Optional[str]]:
    """Collect article URLs and next page link from listing page."""
    soup = get_soup(url)
    if not soup:
        return [], None

    article_tags = soup.find_all('li', class_='clearfix')
    article_urls = [tag.find('a')['href'] for tag in article_tags if tag.find('a') and tag.find('a').get('href')]

    next_page = None
    next_tag = soup.find('span', string="»")
    if next_tag and next_tag.parent and next_tag.parent.has_attr('href'):
        next_page = next_tag.parent['href']

    return article_urls, next_page

def fetch_articles_parallel(urls: List[str]) -> List[dict]:
    """Fetch articles concurrently."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(scrape_article, urls))
    return [article for article in results if article is not None]

def save_as_hf_dataset(data: List[dict]) -> None:
    """Save article data as a Hugging Face dataset."""
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(f"news_{date.today().isoformat()}")

def main():
    base_urls = [
        'https://www.moneycontrol.com/news/business',
        'https://www.moneycontrol.com/news/markets',
        'https://www.moneycontrol.com/news/technology', 
        'https://www.moneycontrol.com/news/personal-finance',
        'https://www.moneycontrol.com/news/commodities',
        'https://www.moneycontrol.com/news/cryptocurrency'
    ]

    all_articles = []
    seen_urls = set()

    for category_url in base_urls:
        print(f"\n[INFO] Scraping category: {category_url}")
        next_page = category_url
        pages_scraped = 0

        while next_page:
            article_urls, next_page = collect_article_links(next_page)
            new_urls = [url for url in article_urls if url not in seen_urls]
            seen_urls.update(new_urls)

            print(f"[INFO] Scraping {len(new_urls)} articles from page {pages_scraped + 1}")
            articles = fetch_articles_parallel(new_urls)
            all_articles.extend(articles)
            pages_scraped += 1

    print(f"\n✅ Total articles scraped: {len(all_articles)}")
    if all_articles:
        save_as_hf_dataset(all_articles)

if __name__ == "__main__":
    main()
