import os
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


class MediaWikiScraper:
    def __init__(self, base_url, output_dir):
        self.base_url = base_url  # Base URL for the MediaWiki website
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def fetch_page_html(self, page_url):
        """
        Fetch the HTML content of a MediaWiki page by scraping the site directly.
        :param page_url: URL of the page to retrieve.
        :return: HTML content of the page.
        """
        full_url = f"{self.base_url}{page_url}"
        print(f"Fetching URL: {full_url}")

        try:
            response = requests.get(full_url)
            # Check if the page exists (200 OK)
            if response.status_code != 200:
                print(f"Failed to fetch page '{full_url}'. Status code: {response.status_code}")
                return None

            return response.text

        except requests.RequestException as e:
            print(f"An error occurred while fetching page '{full_url}': {e}")
            return None

    def extract_content(self, html):
        """
        Extract the main content of the MediaWiki page.
        :param html: HTML content of the page.
        :return: Extracted HTML content of the main body.
        """
        soup = BeautifulSoup(html, 'html.parser')
        content_div = soup.find("div", {"id": "mw-content-text"})

        if content_div:
            return str(content_div)
        else:
            print("Could not find the content on the page.")
            return None

    def save_as_markdown(self, page_title, html_content):
        """
        Save the extracted HTML content as a markdown file.
        :param page_title: Title of the page to use as filename.
        :param html_content: The HTML content to convert and save.
        """
        # Convert HTML to markdown
        markdown_content = md(html_content)

        # Create a valid filename from the page title
        valid_filename = page_title.replace("/", "_") + ".md"

        # Save markdown to file
        filepath = os.path.join(self.output_dir, valid_filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"Saved: {filepath}")

    def extract_all_page_links(self, html):
        """
        Extract all links from the Special:AllPages page.
        :param html: HTML content of the All Pages listing.
        :return: List of URLs to individual pages.
        """
        soup = BeautifulSoup(html, 'html.parser')
        content_div = soup.find("div", {"id": "mw-content-text"})
        page_links = []

        if content_div:
            for link in content_div.find_all("a", href=True):
                href = link['href']
                # Only consider internal wiki links, not external URLs
                if href.startswith("/wiki/") and not ":" in href:
                    page_links.append(href)

        return page_links

    def scrape_all_pages(self):
        """
        Scrape all pages from the MediaWiki site's "Special:AllPages" section.
        """
        print("Fetching all page links...")
        all_pages_url = "/wiki/Special:AllPages"
        all_pages_html = self.fetch_page_html(all_pages_url)

        if all_pages_html:
            page_links = self.extract_all_page_links(all_pages_html)
            print(f"Found {len(page_links)} pages.")

            for page_link in page_links:
                print(f"Scraping page: {page_link}")
                page_html = self.fetch_page_html(page_link)
                if page_html:
                    page_title = page_link.split("/wiki/")[-1].replace("_", " ")  # Extract title from URL
                    content = self.extract_content(page_html)
                    if content:
                        self.save_as_markdown(page_title, content)


# Example usage
if __name__ == "__main__":
    base_url = "https://helpdesk.zcu.cz/"  # Change to the MediaWiki website you want to scrape
    output_dir = "scraped_markdown"  # Directory to save markdown files

    scraper = MediaWikiScraper(base_url, output_dir)
    scraper.scrape_all_pages()
