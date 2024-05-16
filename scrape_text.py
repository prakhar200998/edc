import requests
from bs4 import BeautifulSoup
import re

def scrape_text_from_url(url):
    try:
        # Fetch the content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from specific tags (you can modify this list based on your needs)
        elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])

        # Join the text content from these elements
        text = "\n".join(element.get_text(strip=True) for element in elements)

        # Split text into sentences and handle lists properly
        sentences = re.split(r'(?<=[.!?]) +', text)
        formatted_text = "\n".join(sentences)

        return formatted_text.strip()

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
    url = "https://my.clevelandclinic.org/health/treatments/16859-chemotherapy"

    # Scrape text from the URL
    text = scrape_text_from_url(url)

    if text:
        # Save the text to a file
        save_text_to_file(text, "scraped_text.txt")
