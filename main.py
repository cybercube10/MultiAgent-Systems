from src.tools.tools import web_search,scrape_url

# output = web_search("Latest news on AI ")
results = scrape_url("https://www.nytimes.com/2024/06/05/technology/ai-chatgpt-google.html")
print(results)