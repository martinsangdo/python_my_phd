# %%
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import requests
from bs4 import BeautifulSoup
import pymongo

# %%
db_client = pymongo.MongoClient('mongodb://localhost:27017')
collections = db_client['db_migration']
tb_article = collections['tb_article']

# %%
DOMAIN_URL = 'https://thenewhumanitarian.org'

# %%
def scrape_text_data(url):
    try:
        # Set up Chrome WebDriver
        service = Service('../chromedriver') # or geckodriver for firefox
        options = webdriver.ChromeOptions()
        #options.add_argument('headless')
        options.add_argument("disable-gpu")
        options.add_argument("--window-size=0,0")
        driver = webdriver.Chrome(service=service, options=options) # or webdriver.Firefox(service=service)

        # Navigate to the webpage
        driver.get(url)
        # driver.minimize_window()
        # driver.set_window_size(1, 1)

        # Wait for JavaScript to load (adjust time as needed)
        driver.implicitly_wait(10)  # Waits up to 10 seconds for elements to appear

        # Get the rendered HTML
        html = driver.page_source

        # Close the browser
        driver.quit()

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        return soup
    except:
        print('Error when scraping webpage')
        return ''

# %%
def upsert_article_meta(article_detail):
        db_article = tb_article.find_one({'url': article_detail['url']})
        if db_article is None:
            #insert
            tb_article.insert_one(article_detail)
            # print("Inserted +++++++++++ article: " + article_detail['title'])
        else:
            #update
            tb_article.update_one({'url': article_detail['url']}, {'$set': article_detail})
            # print("Updated ............ article: " + article_detail['title'])

# %%
#There may have different content type
#1: normal article: (https://www.thenewhumanitarian.org/opinion/2025/02/04/how-europe-can-escape-migration-deterrence-trap)
#2: report article: (https://www.thenewhumanitarian.org/analysis/2025/01/07/trends-will-spur-humanitarian-needs-2025)
def extract_content(soup):
    content = ''
    #type 1
    big_content = soup.find('div', attrs={'class': 'field-name-body flow'})
    items = big_content.find_all('p')

    no_of_paragraphs = len(items)
    #print(str(no_of_paragraphs))
    for i in range(0, no_of_paragraphs):
        if i == no_of_paragraphs -1:
            #this is the last item, we need to check whether it is note or not (https://www.thenewhumanitarian.org/opinion/2025/02/04/how-europe-can-escape-migration-deterrence-trap)
            em_tag = items[i].find('em')
            if em_tag is not None:
                break   #do not include this paragraph in the final content
        content += items[i].text.strip()
    #type 2
    items = soup.find_all('div', attrs={'class': 'advanced-report-content flow'})
    no_of_paragraphs = len(items)
    #print(str(no_of_paragraphs))
    for i in range(0, no_of_paragraphs):
        p = items[i].find_all('p')
        for j in range(0, len(p)):
            content += p[j].text.strip()

    return content

# %%
#find article that haven't scraped its content
db_article = tb_article.find_one({'is_scraped': 0})
if db_article is not None:
    url = db_article['url']
    soup = scrape_text_data(DOMAIN_URL + url)
    if soup == '':
        print('Error no soup: ' + url)
        db_article['error'] = 'no soup'
        tb_article.update_one({'url': url}, {'$set': db_article})
    else:
        db_article['is_scraped'] = 1
        content = extract_content(soup)
        if content == '':
            db_article['error'] = 'no content'
            tb_article.update_one({'url': url}, {'$set': db_article})
            print('Error content in page: ' + url)
        else:
            #correct data
            db_article['error'] = ''
            db_article['content'] = content
            db_article['len_content'] = len(content)
            tb_article.update_one({'url': url}, {'$set': db_article})
            # print(content)
            print('Updated content: ' + url)

# %%



