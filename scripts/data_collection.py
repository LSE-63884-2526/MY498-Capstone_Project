import requests
from lxml import html
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "Data"

HEADERS = {
    'User-Agent': 'For academic research purposes (LSE MSc Thesis - CHP Discourse Analysis (November 2023 - Present); contact: e.avci@lse.ac.uk)'
    
}

BASE_URL = 'https://chp.org.tr'

# Beginning of the time frame
# Data collection started on 27 March 2026 23:40
CUTOFF_DATE = datetime(2015, 1, 1)

CATEGORIES = {
    'parliament':      '/gundem/tbmm-konusmalari',
    'rally':           '/gundem/miting-konusmalari',
    'event':           '/gundem/etkinlik-konusmalari',
    'press':           '/gundem/basin-aciklamalari',
    'convention':      '/gundem/kurultay-konusmalari',
}

TV = {'tv': '/gundem/tv-programlari'}
SOCIAL_MEDIA = {'sosyal_medya': '/gundem/sosyal-medya-paylasimlari'}
TBMM_GRUBU = {'tbmm_grubu': '/gundem/tbmm-grubu'}


def detect_speaker(title):
    """Detects if the speech is delivered by Özgür Özel or Kemal Kılıçdaroğlu"""
    if title is None:
        return 'unknown'
    title_lower = title.lower()
    if 'kılıçdaroğlu' in title_lower:
        return 'kilicdaroglu'
    elif 'özgür özel' in title_lower:
        return 'ozel'
    else:
        return 'unknown' # freq = 0


def get_speech_metadata(category_path):
    """Collects speech URLs, dates and titles from pages.
    Stops when it reaches CUTOFF_DATE."""
    speeches = []
    page = 1
    while True:
        listing_url = f"{BASE_URL}{category_path}?page={page}"
        response = requests.get(listing_url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Stopped at page {page} - status {response.status_code}")
            break
        tree = html.fromstring(response.content)
        cards = tree.xpath('//a[contains(@class, "journal-allnews-news-border")]')
        if not cards:
            print(f"No cards found at page {page}")
            break
        stop = False
        for card in cards:
            date_raw = card.xpath('.//p[contains(@class, "date")]//text()')
            title_raw = card.xpath('.//h4//text()')
            href = card.get('href', '')
            if not date_raw:
                continue
            date_str = date_raw[0].strip()
            try:
                date_parsed = datetime.strptime(date_str, '%d.%m.%Y')
            except ValueError:
                continue
            if date_parsed < CUTOFF_DATE:
                print(f"Reached cutoff at {date_str}, stopping.")
                stop = True
                break
            full_url = BASE_URL + href if href.startswith('/') else href
            title = title_raw[0].strip() if title_raw else None
            speeches.append({'url': full_url, 'date': date_str, 'title': title})
        print(f"Page {page}: {len(cards)} cards, {len(speeches)} total so far")
        if stop:
            break
        page += 1
        time.sleep(2)
    return speeches


def scrape_speech_text(url, retries=3, wait=5):
    """Scrapes raw speech text from a single speech using newsDetail div.
    If encounter an error, retries a couple of times."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 404:
                print(f"404 - skipping: {url}")
                return None
            if response.status_code != 200:
                print(f"Attempt {attempt+1}: status {response.status_code} for {url}")
                time.sleep(wait)
                continue
            tree = html.fromstring(response.content)
            content_div = tree.xpath('//*[@id="newsDetail"]/div')
            if not content_div:
                print(f"No content found: {url}")
                return None
            return content_div[0].text_content().strip()
        except Exception as e:
            print(f"Attempt {attempt+1} error on {url}: {e}")
            time.sleep(wait)
    print(f"All retries failed: {url}")
    return None


def collect_corpus(categories, label):
    """Collects speeches for a given set of categories and saves to CSV.
    Speaker is detected from title. Failed URLs are saved separately."""
    speeches = []
    failed_urls = []
    for venue, path in categories.items():
        print(f"\n=== {venue.upper()} ===")
        metadata_list = get_speech_metadata(path)
        print(f"Total to scrape: {len(metadata_list)}")
        for i, meta in enumerate(metadata_list):
            print(f"[{i+1}/{len(metadata_list)}]")
            text = scrape_speech_text(meta['url'])
            speaker = detect_speaker(meta['title'])
            if text:
                speeches.append({
                    'id': f"{venue}_{speaker}_{meta['date'].replace('.', '')}_{i}",
                    'text':    text,
                    'date':    meta['date'],
                    'title':   meta['title'],
                    'url':     meta['url'],
                    'venue':   venue,
                    'speaker': speaker
                })
                print(f"{meta['date']} | {venue} | {speaker} | {meta['title'][:50]}")
            else:
                failed_urls.append({
                    'id': f"{venue}_{speaker}_{meta['date'].replace('.', '')}_{i}",
                    'url':     meta['url'],
                    'date':    meta['date'],
                    'title':   meta['title'],
                    'venue':   venue,
                    'speaker': speaker
                })
            time.sleep(2)
    df = pd.DataFrame(speeches)

    # use utf-8-sig for turkish language
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / f'corpus_{label}.csv', index=False, encoding='utf-8-sig')
    print(f"Saved: corpus_{label}.csv ({df.shape[0]} rows)")

    # recover failed_urls
    if failed_urls:
        df_failed = pd.DataFrame(failed_urls)
        df_failed.to_csv(DATA / f'failed_{label}.csv', index=False, encoding='utf-8-sig')
        print(f"Failed URLs saved: failed_{label}.csv ({len(failed_urls)} rows)")
    return df


if __name__ == '__main__':
    df_main    = collect_corpus(CATEGORIES,   'main')
    df_tv      = collect_corpus(TV,           'tv')
    df_social  = collect_corpus(SOCIAL_MEDIA, 'social_media')
    df_tbmm_gr = collect_corpus(TBMM_GRUBU,  'tbmm_grubu')