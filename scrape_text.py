import requests
from bs4 import BeautifulSoup

def scrape_text_from_url(url):
    try:
        # Fetch the content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Extract the main content
        content = soup.find('div', {'id': 'bodyContent'})
        if content is None:
            content = soup.find('div', {'class': 'mw-body'})
        if content is None:
            content = soup

        # Extract text from the relevant elements
        elements = content.find_all(['h1', 'h2', 'h3', 'p', 'li'])
        text = " ".join(element.get_text(separator=" ", strip=True) for element in elements)

        # Replace non-breaking spaces with regular spaces
        text = text.replace(u'\xa0', ' ')

        return text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

def save_text_to_file(text, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving the text to file: {e}")

if __name__ == "__main__":
    # Example URL
    url = "https://en.wikipedia.org/wiki/Chemotherapy"

    # Scrape text from the URL
    text = scrape_text_from_url(url)

    if text:
        # Save the text to a file
        save_text_to_file(text, "scraped_text.txt")
