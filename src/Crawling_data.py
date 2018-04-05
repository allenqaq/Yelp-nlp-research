
'''
Created on Dec 2, 2017

@author: allen
'''

import urllib
import urllib2
from bs4 import BeautifulSoup
import json
import pickle
import time

def crawling(url):
    nodes = []
    
    html = urllib.urlopen(url).read()
    html = html.decode("utf-8")
    
    soup = BeautifulSoup(html,"lxml")
    js = soup.select_one("script[type=application/ld+json]").text
    js = json.loads(js)
    
    for i in js["review"]:
        node = {}
        node["name"] = js["name"]
#         node["type"] = js["@type"]
        node["ratingValue"] = i["reviewRating"]["ratingValue"]
        node["review"] = i["description"]
        node["review"] = node["review"].replace('\n', '')

        nodes.append(node)
    
    return nodes

def reviews_list(url):
    review_urllist = []
    for i in range(5):
        page_url = url + "?start=" + str(i) +"0"
        review_urllist.append(page_url)
        
    return review_urllist

def restaurant_page_urllist(num):
    restaurant_page_urllist = []
    
    url1 = "https://www.yelp.com/search?find_loc=Washington&start="
    url2 = "0&cflt=restaurants"
    
    num = num/10
    
    for i in range(num):
        url = url1 + str(i) + url2
        restaurant_page_urllist.append(url)
    for i in restaurant_page_urllist:
        print(i)
    return restaurant_page_urllist

def get_restaurant_URL(url):
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    html_doc = response.read()
    
    soup = BeautifulSoup(html_doc,"lxml")  
    print(url)
    review_count = soup.find_all("span", "review-count rating-qualifier")
    review_count_list = []
    for i in review_count:
        s = i.string.strip()
        s = s.split(' ')
        review_count_list.append(s[0])
      
    print(review_count_list)
    
    content = soup.find_all("a", "biz-name js-analytics-click")
    urllist = []
    for i in content:
        urllist.append("https://www.yelp.com"+i["href"])
        
#     print(len(urllist))

    return_list = []
    for i in range(1,11):
        if int(review_count_list[i]) > 180:
            return_list.append(urllist[i])   
    
    return return_list

def collect_review_url():
    print("1.....get_restaurant_URL")
    page_list = restaurant_page_urllist(990)
    url_list = []
    for i in page_list:
        time.sleep(5)
        one_url = get_restaurant_URL(i)
        if one_url:
            url_list.append(one_url)
    print("1....done")
    
    print("2....pages")
    review_url = []
    for i in url_list:
        for j in i:
            one_review_url = reviews_list(j)
            review_url.append(one_review_url)
    print("2.....done")
    print(len(review_url))
    
    return review_url

def collect_review(review_url):
    print("3...start")
    s = 0
    data = []
    for i in review_url:
        for j in i:
            time.sleep(4)
            nodes = crawling(j)
            for k in nodes:
                data.append(k)
#                 print(k)
        s = s + 1
        print("-----------------------------------")
        print(s*100.0/len(review_url))
        print(len(review_url))
        print("-----------------------------------")

#     for i in data:
#         print(i)
#     pickle.dump(data, open("yelp_review_dataset_v2", "w"))
###     data = pickle.load(open("yelp_review_dataset"))
    return data
if __name__ == '__main__':
    start = time.clock()
#     review_url = collect_review_url()
#     pickle.dump(review_url, open("yelp_review_url_dataset_v2", "w"))
    review_url = pickle.load(open("yelp_review_url_dataset_v2"))
    print(review_url)
#     data = collect_review(review_url)
#     pickle.dump(data, open("yelp_review_dataset_v2", "w"))
    print("1---")
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    data1 = collect_review(review_url[:100])
    pickle.dump(data1, open("yelp_review_dataset_v2_data1", "w"))
    time.sleep(300)
    print("2---")
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    data2 = collect_review(review_url[100:200])
    pickle.dump(data2, open("yelp_review_dataset_v2_data2", "w"))
    time.sleep(300)
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    print("3---")
    data3 = collect_review(review_url[200:300])
    pickle.dump(data3, open("yelp_review_dataset_v2_data3", "w"))
    time.sleep(300)
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    print("4---")
    data4 = collect_review(review_url[300:400])
    pickle.dump(data4, open("yelp_review_dataset_v2_data4", "w"))
    time.sleep(300)
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    print("5---")
    data5 = collect_review(review_url[400:])
    pickle.dump(data5, open("yelp_review_dataset_v2_data5", "w"))
    
    data = []
    data.extend(data1)
    data.extend(data2)
    data.extend(data3)
    data.extend(data4)
    data.extend(data5)
    
    pickle.dump(data, open("yelp_review_dataset_v2", "w"))
    
    