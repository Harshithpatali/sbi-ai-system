from newsapi import NewsApiClient
from config import NEWS_API_KEY

print("API KEY:", NEWS_API_KEY)

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

articles = newsapi.get_everything(
    q="SBI",
    language="en",
    page_size=5
)

print(articles)