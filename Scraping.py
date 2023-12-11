import os
import http.server
import socketserver
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import base64

def can_fetch(url, user_agent="*"):
    rp = RobotFileParser()
    rp.set_url(urljoin(url, "/robots.txt"))
    rp.read()
    return rp.can_fetch(user_agent, url)

def download_file(url, folder):
    if url.startswith('data:'):
        data_parts = url.split(',')
        if len(data_parts) == 2:
            data_type, data_base64 = data_parts
            data_base64 += '=' * ((4 - len(data_base64) % 4) % 4)
            extension = data_type.split(';')[0].split('/')[1]
            filename = os.path.join(folder, f'data_url.{extension}')
            with open(filename, 'wb') as file:
                file.write(base64.b64decode(data_base64))
            print(f"Downloaded (Data URL): {filename}")
        else:
            print(f"Failed to download (Invalid data URL): {url}")
    else:
        response = requests.get(url)
        if response.status_code == 200:
            filename = os.path.join(folder, sanitize_filename(os.path.basename(url)))
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")



def sanitize_filename(filename):
    invalid_chars = r'\/:*?"<>|'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    filename = filename.replace('%', '_percent_').replace('&', '_and_').replace('=', '_equals_')

    return filename



def update_html_paths(html_content, output_folder, website_url):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Update CSS links
    css_links = soup.find_all('link', rel='stylesheet')
    for link in css_links:
        link['href'] = os.path.join(output_folder, os.path.basename(urljoin(website_url, link['href']))).replace("\\", "/")

    # Update JavaScript links
    script_tags = soup.find_all('script', src=True)
    for script_tag in script_tags:
        script_tag['src'] = os.path.join(output_folder, os.path.basename(urljoin(website_url, script_tag['src']))).replace("\\", "/")

    # Update image sources
    img_tags = soup.find_all('img', src=True)
    for img_tag in img_tags:
        img_url = urljoin(website_url, img_tag['src'])
        img_filename = os.path.basename(img_url)
        img_path = os.path.join(output_folder, img_filename).replace("\\", "/")

        # Download image or use placeholder if not found
        if not os.path.exists(img_path):
            download_file(img_url, output_folder)
        if not os.path.exists(img_path):
            img_tag['src'] = os.path.join(output_folder, 'placeholder.jpg').replace("\\", "/")
        else:
            img_tag['src'] = os.path.join(output_folder, img_filename).replace("\\", "/")

    return str(soup)

def download_website(url, base_output_folder, html_output_folder):
    print("Starting " + url)
    
    # Step 1: Download HTML
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return

    # Step 2: Parse HTML and download linked resources (CSS and JS)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Download CSS files
    css_links = soup.find_all('link', rel='stylesheet')
    for link in css_links:
        css_url = urljoin(url, link.get('href'))
        download_file(css_url, base_output_folder)

    # Download JavaScript files
    script_tags = soup.find_all('script', src=True)
    for script_tag in script_tags:
        js_url = urljoin(url, script_tag['src'])
        download_file(js_url, base_output_folder)

    # Save the HTML content after updating paths
    updated_html_content = update_html_paths(html_content, base_output_folder, url)
    website_name = get_website_name(url)
    html_filename = f'{website_name.replace(".", "_")}.html'
    html_output_path = os.path.join(html_output_folder, html_filename)
    with open(html_output_path, 'w', encoding='utf-8') as html_file:
        html_file.write(updated_html_content)

    # Step 1: Download HTML
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return

    # Step 2: Parse HTML and download linked resources (CSS and JS)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Download CSS files
    css_links = soup.find_all('link', rel='stylesheet')
    for link in css_links:
        css_url = urljoin(url, link.get('href'))
        download_file(css_url, base_output_folder)

    # Download JavaScript files
    script_tags = soup.find_all('script', src=True)
    for script_tag in script_tags:
        js_url = urljoin(url, script_tag['src'])
        download_file(js_url, base_output_folder)

    # Save the HTML content after updating paths
    updated_html_content = update_html_paths(html_content, base_output_folder, url)
    website_name = get_website_name(url)
    html_filename = f'{website_name.replace(".", "_")}.html'
    with open(os.path.join('', html_filename), 'w', encoding='utf-8') as html_file:
        html_file.write(updated_html_content)
    print("Done with " + url)


def serve_website(port):
    web_dir = os.path.join(os.getcwd(), '')

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            # Override to use the 'output' directory
            return os.path.abspath(os.path.join(web_dir, path.lstrip('/')))

    Handler = CustomHandler

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving website at http://localhost:{port}")
        httpd.serve_forever()
        
def get_website_name(url):
    domain_parts = url.split("//")[-1].split("/")[0].split(".")

    if domain_parts[0] == 'www':
        return '.'.join(domain_parts[1:])
    else:
        return '.'.join(domain_parts)

def process_websites(websites):
    for website_url in websites:
        try:
            if not can_fetch(website_url):
                print(f"Skipping {website_url}. Not allowed by robots.txt.")
                continue

            website_name = get_website_name(website_url)
            output_directory = website_name

            if os.path.exists(output_directory):
                print(f"Skipping {website_url}. Folder '{output_directory}' already exists.")
                continue

            os.mkdir(output_directory)
            download_website(website_url, output_directory, '')

        except Exception as e:
            print(f"Error processing {website_url}: {e}")
            continue



website_urls = [
    'https://books.toscrape.com/',
    'https://www.wikipedia.org/',
    'https://www.python.org/',
    'https://www.nytimes.com/',
    'https://www.nationalgeographic.com/',
    'https://www.space.com/',
    'https://www.nike.com/',
    'https://www.apple.com/',
    'https://www.microsoft.com/',
    'https://www.cnn.com/',
    'https://www.bbc.com/news',
    'https://www.weather.com/',
    'https://www.imdb.com/',
    'https://www.github.com/',
    'https://www.stackoverflow.com/',
    'https://www.reddit.com/',
    'https://www.quora.com/',
    'https://www.linkedin.com/',
    'https://www.instagram.com/',
    'https://www.pexels.com/',
    'https://unsplash.com/',
    'https://www.flickr.com/',
    'https://www.github.com/',
    'https://www.bitbucket.org/',
    'https://www.gitlab.com/',
    'https://www.netflix.com/',
    'https://www.amazon.com/',
    'https://www.ebay.com/',
    'https://www.etsy.com/',
    'https://www.walmart.com/',
    'https://www.target.com/',
    'https://www.apple.com/itunes/',
    'https://www.spotify.com/',
    'https://www.soundcloud.com/',
    'https://www.last.fm/',
    'https://www.youtube.com/',
    'https://www.vimeo.com/',
    'https://www.dailymotion.com/',
    'https://www.ted.com/',
    'https://www.coursera.org/',
    'https://www.udacity.com/',
    'https://www.edx.org/',
    'https://www.khanacademy.org/',
    'https://www.brainly.com/',
    'https://www.medium.com/',
    'https://www.blogger.com/',
    'https://www.wordpress.com/',
    'https://www.tumblr.com/',
    'https://www.buzzfeed.com/',
    'https://www.huffpost.com/',
    'https://www.forbes.com/',
    'https://www.techcrunch.com/',
    'https://www.wired.com/',
    'https://www.cnbc.com/',
    'https://www.bloomberg.com/',
    'https://www.businessinsider.com/',
    'https://www.nike.com/',
    'https://www.puma.com/',
    'https://www.underarmour.com/',
    'https://www.zara.com/',
    'https://www.hm.com/',
    'https://www.forever21.com/',
    'https://www.ikea.com/',
    'https://www.homedepot.com/',
    'https://www.walmart.com/',
    'https://www.target.com/',
    'https://www.microsoftstore.com/',
    'https://www.apple.com/',
    'https://www.samsung.com/',
    'https://www.lg.com/',
    'https://www.sony.com/',
    'https://www.panasonic.com/',
    'https://www.philips.com/',
    'https://www.nintendo.com/',
    'https://www.playstation.com/',
    'https://www.xbox.com/',
    'https://www.steamcommunity.com/',
    'https://www.origin.com/',
    'https://www.uplay.com/',
    'https://www.epicgames.com/',
    'https://www.itch.io/',
    'https://www.gog.com/',
    'https://www.twitch.tv/',
    'https://www.youtube.com/',
    'https://www.facebook.com/',
    'https://www.mixer.com/',
    'https://www.cbs.com/',
    'https://www.nbc.com/',
    'https://www.abc.com/',
    'https://www.fox.com/',
    'https://www.cw.com/',
    'https://www.cartoonnetwork.com/',
    'https://www.nick.com/',
    'https://www.disneyplus.com/',
    'https://www.netflix.com/',
    'https://www.hulu.com/',
    'https://www.amazon.com/',
    'https://www.apple.com/',
    'https://www.hbomax.com/',
    'https://www.sling.com/',
    'https://www.crunchyroll.com/',
    'https://www.funimation.com/',
    'https://www.vrv.co/',
    'https://www.tidal.com/',
    'https://www.deezer.com/',
    'https://www.pandora.com/',
    'https://www.spotify.com/',
    'https://www.apple.com/music/',
    'https://www.amazon.com/music/',
    'https://www.google.com/maps/',
    'https://www.apple.com/maps/',
    'https://www.microsoft.com/maps/',
    'https://www.yahoo.com/maps/',
    'https://www.bing.com/maps/',
    'https://www.waze.com/',
    'https://www.tomtom.com/',
    'https://www.mapquest.com/',
    'https://www.openstreetmap.org/',
    'https://www.strava.com/',
    'https://www.runkeeper.com/',
    'https://www.myfitnesspal.com/',
    'https://www.fitbit.com/',
    'https://www.underarmour.com/',
    'https://www.nike.com/',
    'https://www.puma.com/',
    'https://www.myprotein.com/',
    'https://www.bodybuilding.com/',
    'https://www.gnc.com/',
    'https://www.iherb.com/',
    'https://www.amazon.com/',
    'https://www.ebay.com/',
    'https://www.walmart.com/',
    'https://www.target.com/',
    'https://www.microsoftstore.com/',
    'https://www.apple.com/',
    'https://www.samsung.com/',
    'https://www.lg.com/',
    'https://www.sony.com/',
    'https://www.panasonic.com/',
    'https://www.philips.com/',
    'https://www.nintendo.com/',
    'https://www.playstation.com/',
    'https://www.xbox.com/',
    'https://www.steamcommunity.com/',
    'https://www.origin.com/',
    'https://www.uplay.com/',
    'https://www.epicgames.com/',
    'https://www.itch.io/',
    'https://www.gog.com/',
    'https://www.twitch.tv/',
    'https://www.youtube.com/gaming',
    'https://www.facebook.com/gaming',
    'https://www.mixer.com/',
    'https://www.cbs.com/',
    'https://www.nbc.com/',
    'https://www.abc.com/',
    'https://www.fox.com/',
    'https://www.cw.com/',
    'https://www.cartoonnetwork.com/',
    'https://www.nick.com/',
    'https://www.disneyplus.com/',
    'https://www.netflix.com/',
    'https://www.hulu.com/',
    'https://www.amazon.com/primevideo/',
    'https://www.apple.com/tv/',
    'https://www.hbomax.com/',
    'https://www.sling.com/',
    'https://www.crunchyroll.com/',
    'https://www.funimation.com/',
    'https://www.vrv.co/',
    'https://www.tidal.com/',
    'https://www.deezer.com/',
    'https://www.pandora.com/',
    'https://www.spotify.com/',
    'https://www.apple.com/music/',
    'https://www.amazon.com/music/',
    'https://www.google.com/maps/',
    'https://www.apple.com/maps/',
    'https://www.microsoft.com/maps/',
    'https://www.yahoo.com/maps/',
    'https://www.bing.com/maps/',
    'https://www.waze.com/',
    'https://www.tomtom.com/',
    'https://www.mapquest.com/',
    'https://www.openstreetmap.org/',
    'https://www.strava.com/',
    'https://www.runkeeper.com/',
    'https://www.myfitnesspal.com/',
    'https://www.fitbit.com/',
    'https://www.underarmour.com/',
    'https://www.nike.com/',
    'https://www.puma.com/',
    'https://www.myprotein.com/',
    'https://www.bodybuilding.com/',
    'https://www.gnc.com/',
    'https://www.iherb.com/',
    'https://www.amazon.com/',
    'https://www.ebay.com/',
    'https://www.walmart.com/',
    'https://www.microsoftstore.com/',
    'https://www.apple.com/',
    'https://www.samsung.com/',
    'https://www.lg.com/',
    'https://www.sony.com/',
    'https://www.panasonic.com/',
    'https://www.philips.com/',
    'https://www.nintendo.com/',
    'https://www.playstation.com/',
    'https://www.xbox.com/',
    'https://www.steamcommunity.com/',
    'https://www.origin.com/',
    'https://www.uplay.com/',
    'https://www.epicgames.com/',
    'https://www.itch.io/',
    'https://www.gog.com/',
    'https://www.twitch.tv/',
    'https://www.youtube.com/gaming',
    'https://www.facebook.com/gaming',
    'https://www.mixer.com/',
    'https://www.cbs.com/',
    'https://www.nbc.com/',
    'https://www.abc.com/',
    'https://www.fox.com/',
    'https://www.cw.com/',
    'https://www.cartoonnetwork.com/',
    'https://www.nick.com/',
    'https://www.disneyplus.com/',
    'https://www.netflix.com/',
    'https://www.hulu.com/',
    'https://www.amazon.com/primevideo/',
    'https://www.apple.com/tv/',
    'https://www.hbomax.com/',
    'https://www.sling.com/',
    'https://www.crunchyroll.com/',
    'https://www.funimation.com/',
    'https://www.vrv.co/',
    'https://www.tidal.com/',
    'https://www.deezer.com/',
    'https://www.pandora.com/',
    'https://www.spotify.com/',
    'https://www.apple.com/music/',
    'https://www.amazon.com/music/',
    'https://www.google.com/maps/',
    'https://www.apple.com/maps/',
    'https://www.microsoft.com/maps/',
    'https://www.yahoo.com/maps/',
    'https://www.bing.com/maps/',
    'https://www.waze.com/',
    'https://www.tomtom.com/',
    'https://www.mapquest.com/',
    'https://www.openstreetmap.org/',
    'https://www.strava.com/',
    'https://www.runkeeper.com/',
    'https://www.myfitnesspal.com/',
    'https://www.fitbit.com/',
    'https://www.underarmour.com/',
    'https://www.nike.com/',
    'https://www.puma.com/',
    'https://www.myprotein.com/',
    'https://www.bodybuilding.com/',
    'https://www.gnc.com/',
    'https://www.iherb.com/',
    'https://www.amazon.com/',
    'https://www.ebay.com/',
    'https://www.walmart.com/',
    'https://www.target.com/',
    'https://www.microsoftstore.com/',
    'https://www.apple.com/',
    'https://www.samsung.com/',
    'https://www.lg.com/',
    'https://www.sony.com/',
    'https://www.panasonic.com/',
    'https://www.philips.com/',
    'https://www.nintendo.com/',
    'https://www.playstation.com/',
    'https://www.xbox.com/',
    'https://www.steamcommunity.com/',
    'https://www.origin.com/',
    'https://www.uplay.com/',
    'https://www.epicgames.com/',
    'https://www.itch.io/',
    'https://www.gog.com/',
    'https://www.twitch.tv/',
    'https://www.youtube.com/gaming',
    'https://www.facebook.com/gaming',
    'https://www.mixer.com/',
    'https://www.cbs.com/',
    'https://www.nbc.com/',
    'https://www.abc.com/',
    'https://www.fox.com/',
    'https://www.cw.com/',
    'https://www.cartoonnetwork.com/',
    'https://www.nick.com/',
    'https://www.disneyplus.com/',
    'https://www.netflix.com/',
    'https://www.hulu.com/',
    'https://www.amazon.com/primevideo/',
    'https://www.apple.com/tv/',
    'https://www.hbomax.com/',
    'https://www.sling.com/',
    'https://www.crunchyroll.com/',
    'https://www.funimation.com/',
    'https://www.vrv.co/',
    'https://www.tidal.com/',
    'https://www.deezer.com/']

process_websites(website_urls)