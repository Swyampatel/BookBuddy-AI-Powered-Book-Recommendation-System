import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

# Web Scraping to get books data
def scrape_books():
    url = "https://www.goodreads.com/list/show/1.Best_Books_Ever"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError("Failed to fetch the Goodreads page. Check the URL or your internet connection.")

    soup = BeautifulSoup(response.content, 'html.parser')

    books = []
    for book_item in soup.select(".bookTitle")[:50]:  # Limit to 50 books
        title = book_item.text.strip()
        link = "https://www.goodreads.com" + book_item['href']
        books.append({"title": title, "link": link})

    authors = [author.text.strip() for author in soup.select(".authorName")[:50]]

    def fetch_description(book):
        try:
            book_response = requests.get(book["link"], headers=headers)
            book_soup = BeautifulSoup(book_response.content, 'html.parser')
            description = book_soup.select_one("#description span:nth-of-type(2)")
            return description.text.strip() if description else "Description unavailable"
        except Exception:
            return "Description unavailable"

    with ThreadPoolExecutor(max_workers=10) as executor:
        descriptions = list(executor.map(fetch_description, books))

    data = pd.DataFrame({
        "title": [book["title"] for book in books],
        "authors": authors,
        "description": descriptions
    })

    return data

# Load or scrape data
cache_path = Path("scraped_books.csv")
if cache_path.exists():
    data = pd.read_csv(cache_path)
else:
    data = scrape_books()
    data.to_csv(cache_path, index=False)

# Preprocessing: Fill NaNs and combine relevant text fields
data['title'] = data['title'].fillna('').astype(str)
data['authors'] = data['authors'].fillna('').astype(str)
data['description'] = data['description'].fillna('').astype(str)
data['text'] = data['title'] + ' ' + data['authors'] + ' ' + data['description']

# Remove rows with empty 'text'
data = data[data['text'].str.strip() != '']

# Ensure at least one valid row remains
if data.empty:
    raise ValueError("No valid data to process. Please check the scraped content.")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)  # Limit to 5000 features
text_matrix = vectorizer.fit_transform(data['text'])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(text_matrix, text_matrix)

# Console-based interaction
print("BookBuddy Recommendation System is running!")
while True:
    book_title = input("Enter a book title to get recommendations (or type 'exit' to quit): ").strip()
    if book_title.lower() == 'exit':
        print("Exiting the recommendation system. Goodbye!")
        break

    if book_title not in data['title'].values:
        print("Book not found in dataset. Please try another title.")
        continue

    # Find index of the given book
    idx = data[data['title'] == book_title].index[0]

    # Get similarity scores for the book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar books (excluding the book itself)
    top_books = sim_scores[1:6]
    recommendations = [data.iloc[i[0]]['title'] for i in top_books]

    print("Top 5 recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
