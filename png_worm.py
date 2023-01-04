import os, time, requests

def png_downloader(count = 50000):
    # image count
    png_counter = 0

    # request delay, prevent ip from banned
    delay = 0.2

    # check if there are some images downloaded already last time
    for p, d, f in os.walk('captcha/'):
        png_counter = len(f)
        break

    # website grabber, customize session
    worm = requests.Session()
    worm.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54'
    
    # ex: url = 'https://example.com'
    url = '### YOUR TARGET CAPTCHA IMAGE URL ###'

    # download captcha image untill 'count'
    while png_counter < count:
        if png_counter % 100 == 0:
            print('Now: {}'.format(png_counter + 1))
        # prevent same cookies request too many times
        worm.cookies.clear()
        with open('captcha/captcha{}.png'.format(png_counter), 'wb+') as captcha_png:
            captcha_png.write(worm.get(url=url, stream=True).content)
        png_counter += 1
        # prevent ip from banned
        time.sleep(delay)

if __name__ == '__main__':
    png_downloader(150000)