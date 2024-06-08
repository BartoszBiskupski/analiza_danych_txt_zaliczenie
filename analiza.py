import requests
import json

api_key = "8c3d9a03d361452d90d51c841d1a7098"
keyword = "granica"

url = f"https://newsapi.org/v2/top-headlines?country=pl&apiKey={api_key}"

response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()

    # Extract the articles
    articles = data['articles']

    # Print the title and description of each article
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print("\n")
else:
    print(f"Failed to get data from API, status code: {response.status_code}")