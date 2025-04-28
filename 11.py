from random import *
# from module name import modules as a(a== module use "호출")
import requests
from bs4 import BeautifulSoup
class WebScraping:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
    
    def scrape_and_save(self, filename):
        try:
            res = requests.get(self.url, headers=self.headers)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "lxml")
            
            # Extract all text from the page
            text = soup.get_text(separator='\n', strip=True)
            
            # Save the text to a file
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(text)
            
            print(f"Text content saved to {filename}")
        
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")

url = 'https://www.yna.co.kr/view/AKR20250207151000004?section=society/all&site=major_news01'
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
}
scraper = WebScraping(url, headers)
scraper.scrape_and_save('output.txt')


