import requests, time, os
from bs4 import BeautifulSoup

def ask_getty(actor, folder, limit):

    if not os.path.exists('/home/ubuntu/screenshots/train/%s/' % folder):
        os.makedirs('/home/ubuntu/screenshots/train/%s/' % folder)
        os.makedirs('/home/ubuntu/screenshots/validation/%s/' % folder)

    page_limit = limit/100
    
    # scroll through the pages
    for i in range(1, page_limit+1):
        url = 'https://api.gettyimages.com/v3/search/images?fields=id,title,thumb,referral_destinations&compositions=headshot&phrase=%s&page_size=100&page=%d' % (actor, i)
        response = requests.get(url, headers={'Api-Key': '27j5acyxq54nkmtv8smmp577'})
        page = response.json()
        
        # break out of loop once we reach non-existent page
        try:
            imagelist = page['images']
        except:
            break
        
        # get individual images on each page
        for idx, im in enumerate(imagelist, 1):
            with open('/home/ubuntu/screenshots/train/%s/%s%d.jpg' % (folder, folder, idx + 100*(i-1)), 'wb') as f:
                image = requests.get(im['display_sizes'][0]['uri'])
                f.write(image.content)
        if i % 5 == 0:
            print "Finished Iteration %d" % (100*i)
        time.sleep(1)
    
    print "All Finished!"

# ask_getty('christian bale', 'bale', 1000)
# ask_getty('keanu reeves', 'keanu', 1000)
# ask_getty('leonardo dicaprio', 'leo', 1000)
# ask_getty('scarlett johannson', 'scarlett', 1000)
# ask_getty('jennifer lawrence', 'jl', 1000)
# ask_getty('angelina jolie', 'ag', 1000)
# ask_getty('emma stone', 'es', 1000)
# ask_getty('natalie portman', 'np', 1000)
# ask_getty('harrison ford', 'hf', 1000)
# ask_getty('tom cruise', 'tc', 1000)

def get_imdb(url, folder):
    picture_links = []
    for i in range(1,28):
        new_url = url + "page=%d" % i
        response = requests.get(new_url)
        page = response.text
        soup = BeautifulSoup(page, 'lxml')

        links = soup.find_all("a", {"itemprop": "thumbnailUrl"})
        
        if not links:
            break

        for node in links: 
            link = node.img['src']
            picture_links.append(link)
        time.sleep(1)
    print "got all the links for %s" % folder

    for idx, im in enumerate(picture_links, 1):
        with open('/home/ubuntu/screenshots/train/%s/imdb_%s%d.jpg' % (folder, folder, idx), 'wb') as f:
            image = requests.get(im)
            f.write(image.content)
        if idx % 50 == 0:
            print "Finished Iteration %d" % idx
            time.sleep(1)
    print "downloaded all pics for %s" % folder

get_imdb("http://www.imdb.com/name/nm0000288/mediaindex?", 'bale')
get_imdb("http://www.imdb.com/name/nm0000206/mediaindex?", 'keanu')
get_imdb("http://www.imdb.com/name/nm0000138/mediaindex?", 'leo')
get_imdb("http://www.imdb.com/name/nm0424060/mediaindex?", 'scarlett')
get_imdb("http://www.imdb.com/name/nm2225369/mediaindex?", 'jl')
get_imdb("http://www.imdb.com/name/nm0001401/mediaindex?", 'ag')
get_imdb("http://www.imdb.com/name/nm1297015/mediaindex?", 'es')
get_imdb("http://www.imdb.com/name/nm0000204/mediaindex?", 'np')
get_imdb("http://www.imdb.com/name/nm0000148/mediaindex?", 'hf')
get_imdb("http://www.imdb.com/name/nm0000129/mediaindex?", 'tc')
