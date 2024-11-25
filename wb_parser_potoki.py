import requests
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

csv_file = "wb.csv"


def append_to_csv(file_path, data, headers=None):
    file_exists = False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(file_path, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists and headers:
            writer.writeheader()
        writer.writerow(data)

chrome_options = Options()
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless=new")


def scrape_category(category):
    shard = category['shard']
    url = category['url']
    name = category['name']

    ids = []
    photos = []

    driver = webdriver.Chrome(options=chrome_options)

    try:
        for page in range(1, 2):
            driver.get(f'https://www.wildberries.ru{url}?sort=popular&page={page}')
            time.sleep(6 if page == 1 else 3)
            for _ in range(9):
                driver.execute_script("window.scrollBy(0, 1200)")
                time.sleep(1)

            products_data = driver.find_elements(By.TAG_NAME, 'article')
            imgs_data = driver.find_elements(By.CLASS_NAME, 'j-thumbnail')

            ids += [src.get_attribute("data-nm-id") for src in products_data]
            photos += [img.get_attribute("src") for img in imgs_data]

            if len(ids) != len(photos):
                continue

        for h in range(len(ids)):
            try:
                driver.get(f'https://www.wildberries.ru/catalog/{ids[h]}/detail.aspx')
                time.sleep(4)
                try:
                    name = driver.find_element(By.CLASS_NAME, 'product-page__title').text
                except:
                    continue
                driver.execute_script("window.scrollBy(0, 8000)")
                time.sleep(4)
                photos_elems = driver.find_elements(By.CLASS_NAME, 'j-thumbnail')
                if len(photos_elems) == 0:
                    continue
                photos_similar = [src.get_attribute("src") for src in photos_elems]
                photo = photos[h]
                product_data = {
                    'id': ids[h],
                    'name': name,
                    'category': url.replace('/catalog/', ''),
                    'image': photo,
                    'similar': photos_similar
                }
                append_to_csv(csv_file, product_data, headers=["id", "name", "category", "image", "similar"])
            except Exception as e:
                print(f"Error processing product {ids[h]}: {e}")
                continue
        print('Finished -',url)
    except Exception as e:
        print(f"Error scraping category {name}: {e}")
    finally:
        driver.quit()


def main():
    # Load categories from the API
    data = requests.get('https://static-basket-01.wbbasket.ru/vol0/data/main-menu-ru-ru-v3.json').json()
    categories = []

    for i in range(4, 29):
        try:
            for q in data[i]['childs']:
                try:
                    categories.extend(q['childs'])
                except KeyError:
                    categories.append(q)
        except KeyError:
            categories.append(data[i])
    categories = categories[56:]

    print(f"Total categories to process: {len(categories)}")


    max_threads = 10
    with ThreadPoolExecutor(max_threads) as executor:
        futures = []
        for category in categories:
            time.sleep(2)
            executor.submit(scrape_category, category)


        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in thread: {e}")


if __name__ == "__main__":
    main()
