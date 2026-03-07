#Prac 1A: Web Scraping and fetching contents like table from single page
import pandas as pd
import requests
from bs4 import BeautifulSoup
url = "https://www.worldometers.info/world-population/population-by-country/"
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")
table = soup.find("table")
data = []
for row in table.find_all("tr"):
    cols = row.find_all(["th","td"])
    cols = [c.text.strip() for c in cols]
    data.append(cols)
# First row as column names
df = pd.DataFrame(data[1:], columns=data[0])
df.head()

#Prac 1B: Web Scraping and fetching contents from multiple pages
#Code 1:
import requests
from bs4 import BeautifulSoup
for i in range(1,6):
    url = f"https://quotes.toscrape.com/page/{i}/"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")
    for q in quotes:
        text = q.find("span", class_="text").text
        author = q.find("small", class_="author").text
        print(text, "-", author)
print("Done")

#Code 2:
import requests
from bs4 import BeautifulSoup

for i in range(1,6):
    soup = BeautifulSoup(requests.get(f"https://quotes.toscrape.com/page/{i}/").text, "html.parser")

    for q in soup.find_all("div", class_="quote"):
        print(q.find("span", class_="text").text, "-", q.find("small", class_="author").text)

print("Done")
