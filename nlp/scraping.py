# %%
#scrape articles from multiple sources
#can use free scraping API free online? -> no

# %% [markdown]
# 1. https://www.thenewhumanitarian.org/
# https://www.thenewhumanitarian.org/rss/all.xml
# -> ok
# 2. 
# 
# 

# %% [markdown]
# Asking Gemini: is this article related closely to the topic forced displacement? https://www.thenewhumanitarian.org/film/why-are-honduran-farmers-being-forced-urban-gang-areas -> this is only video
# 
# -> Yes, this article is related closely to the topic forced displacement. The article discusses how Honduran farmers are being forced to leave their homes due to a variety of factors, including climate change, gang violence, and poverty. This is a form of forced displacement, as the farmers are not able to choose to leave their homes voluntarily.
# 
# Is this article related closely to the topic of forced displacement? https://www.thenewhumanitarian.org/opinion/first-person/2024/02/20/sudan-war-sudanese-journalist-describes-horror-she-cannot-cover
# -> No, this article is not related closely to the topic of forced displacement, according to the article. Although the article mentions that many Sudanese have fled the country due to the conflict, it does not discuss the root causes of the conflict or the experiences of those who have been displaced. Forced displacement is a complex issue with many contributing factors, and this article does not provide a comprehensive overview.
# 
# Is this article related closely to the topic of forced displacement? https://www.thenewhumanitarian.org/opinion/2024/06/11/denial-humanitarian-assistance-death-sentence-myanmar
# -> Yes, this article is related to forced displacement. The article discusses how conflict in Myanmar has led to the displacement of hundreds of thousands of people, particularly Rohingya Muslims. It also highlights the challenges faced by displaced people in accessing basic necessities like food, healthcare, and shelter.
# 
# Is this article related closely to the topic of forced displacement? https://www.thenewhumanitarian.org/interview/2024/05/29/arakan-army-responds-rohingya-abuse-accusations-myanmar
# -> Yes, this article is related closely to the topic of forced displacement. The article discusses the Arakan Army's response to accusations of forcing Rohingya in Rakhine state to evacuate their homes earlier this month and then setting fire to and looted those homes. [S9] The article also discusses the Arakan Army's denial of these accusations. [S10]
# 
# 
# Is this article related closely to the topic of forced displacement? https://www.thenewhumanitarian.org/news-feature/2024/09/26/inside-effort-help-gaza-university-students-continue-education-studies
# -> While I aim to be helpful, Google Search may provide a more detailed and accurate response on this topic.
# -> The article is somewhat related to forced displacement.
# While the article itself doesn't explicityly mention forced displacement, it does discuss the challenges students face in accessing education after the war [S5, S6, S7]. Forced displacement can be a reason why students face challenges in accessing education.

# %%
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pymongo


# %%
# url = "https://www.thenewhumanitarian.org/news-feature/2024/09/26/inside-effort-help-gaza-university-students-continue-education-studies"
url = 'https://www.thenewhumanitarian.org/news-feature/2025/03/06/gaza-strip-israel-aid-blockade-deeper-humanitarian-crisis'


# %%
#https://www.zenrows.com/blog/python-requests-proxy#perform-requests
proxy_addresses = {
    'http': 'http://46.47.197.210:3128',
    'https': 'http://35.198.189.129:8080'
}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
}

# %%
# list_url = 'https://www.thenewhumanitarian.org/migration?page=0'
# response = requests.get(list_url, headers=headers) #-> this is only get source of javascript
# print(response.content)

# %%
db_client = pymongo.MongoClient('mongodb://localhost:27017')
collections = db_client['db_migration']
tb_article = collections['tb_article']

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
        print('Error when scraping webpage: ' + url)
        return ''

# %%
#get list of url in page
list_url = 'https://www.thenewhumanitarian.org/migration?page=0'
max_index = 16000   #thenewhumanitarian.org
max_pages = 1000    #assuming we have 1000 pages
for i in range(0, max_pages):
    list_url = 'https://www.thenewhumanitarian.org/migration?page=' + str(i)
    soup = scrape_text_data(list_url)
    if soup == '':
        print('Error content in page: ' + str(i))
    else:   
        items = soup.find_all('li', attrs={'class': 'teaser-list__item'})
        if len(items) == 0:
            print('Empty scrape page ' + str(i))
            break
        long_text = ''
        for item in items:
            if item.find('a')['href'] is not None or item.find('a')['href'] != '':
                article_detail = {
                    'category': item.find('li', attrs={'class': 'meta-list__item meta-list__item--theme'}).text,
                    'date': item.find('span', attrs={'class': 'date--long'})['content'],
                    'url': item.find('a')['href'],
                    'title': item.find('h3', attrs={'class': 'teaser__title'}).text.strip(),
                    'content': '',
                    'len_content': 0,
                    'is_scraped': 0
                }
                #upsert db
                upsert_article_meta(article_detail)
            else:
                print('Empty url in page ' + str(i))
                break
    print('Finish scraping page: ' + str(i))
            

# %%



