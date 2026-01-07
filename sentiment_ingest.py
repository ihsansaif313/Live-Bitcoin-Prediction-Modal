"""
Bitcoin Sentiment Ingestion Script
Streams news/social data, scores sentiment, and aggregates for prediction models.
"""

import os
import sys
import time
import yaml
import hashlib
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
from typing import Iterator, Dict, List, Optional
import subprocess

# --- Auto-Install Dependencies ---
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import feedparser
except ImportError:
    print("Installing feedparser...")
    install("feedparser")
    import feedparser

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Installing vaderSentiment...")
    install("vaderSentiment")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from langdetect import detect
except ImportError:
    print("Installing langdetect...")
    install("langdetect")
    from langdetect import detect


# --- Configuration ---
def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
SENTIMENT_CONFIG = config.get('sentiment', {})
KEYWORDS = SENTIMENT_CONFIG.get('keywords', ['Bitcoin', 'BTC'])
LOG_FILE = os.path.join(config.get('paths', {}).get('logs_dir', 'logs'), "sentiment_ingest.log")
EVENTS_CSV = "sentiment_events.csv"
MINUTE_CSV = "sentiment_minute.csv"

# --- Logging ---
if not os.path.exists(os.path.dirname(LOG_FILE)): map(os.makedirs, [os.path.dirname(LOG_FILE)])
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)


# --- Deduplication ---
seen_hashes = set()

def get_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def is_duplicate(text: str) -> bool:
    h = get_hash(text)
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    # Maintain set size to prevent memory leak
    if len(seen_hashes) > 10000:
        seen_hashes.pop()
    return False


# --- Source Adapters ---

class SourceAdapter(ABC):
    @abstractmethod
    def fetch(self) -> Iterator[Dict]:
        """Yields dicts with keys: source, text, time, url (optional)"""
        pass

class RSSSourceAdapter(SourceAdapter):
    def __init__(self, feeds: List[str]):
        self.feeds = feeds

    def fetch(self) -> Iterator[Dict]:
        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # Parse time
                    if hasattr(entry, 'published_parsed'):
                        dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), timezone.utc)
                    elif hasattr(entry, 'updated_parsed'):
                        dt = datetime.fromtimestamp(time.mktime(entry.updated_parsed), timezone.utc)
                    else:
                        dt = datetime.now(timezone.utc)

                    text = f"{entry.title} {entry.description}" if hasattr(entry, 'description') else entry.title
                    
                    yield {
                        'source': 'RSS',
                        'text': text,
                        'time': dt,
                        'url': entry.link if hasattr(entry, 'link') else ''
                    }
            except Exception as e:
                logger.error(f"RSS Error ({url}): {e}")

class TwitterSourceAdapter(SourceAdapter):
    # Placeholder for future implementation
    def fetch(self) -> Iterator[Dict]:
        return iter([])


# --- Processing Logic ---

analyzer = SentimentIntensityAnalyzer()

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False # Fail safe

def calculate_relevance(text, keywords):
    """Simple keyword density scoring"""
    text_lower = text.lower()
    score = 0
    for k in keywords:
        if k.lower() in text_lower:
            score += 1
    # Normalized score (capped at 1.0 for >3 matches)
    return min(score / 3.0, 1.0)

def score_sentiment(event: Dict) -> Dict:
    text = event['text']
    
    # 1. Relevance Check
    rel_score = calculate_relevance(text, KEYWORDS)
    if rel_score < 0.1:
        return None  # Irrelevant
    
    # 2. Language Check
    if not is_english(text):
        return None

    # 3. Sentiment Scoring
    scores = analyzer.polarity_scores(text)
    
    event['sentiment_compound'] = max(min(scores['compound'], 1.0), -1.0) # Clip
    event['sentiment_neg'] = scores['neg']
    event['sentiment_pos'] = scores['pos']
    event['sentiment_neu'] = scores['neu']
    event['relevance_score'] = rel_score
    
    return event


# --- Data Persistence ---

def append_event(row: Dict, path: str):
    df_row = pd.DataFrame([row])
    # Ensure columns order/existence
    cols = ['time', 'source', 'text', 'sentiment_compound', 'sentiment_neg', 'relevance_score', 'url']
    for c in cols:
        if c not in df_row.columns: df_row[c] = ''
            
    hdr = not os.path.exists(path)
    df_row[cols].to_csv(path, mode='a', header=hdr, index=False)

def init_csvs():
    """Ensure CSV files exist with proper headers."""
    if not os.path.exists(EVENTS_CSV):
        cols = ['time', 'source', 'text', 'sentiment_compound', 'sentiment_neg', 'relevance_score', 'url']
        pd.DataFrame(columns=cols).to_csv(EVENTS_CSV, index=False)
        logger.info(f"Initialized {EVENTS_CSV}")
    if not os.path.exists(MINUTE_CSV):
        cols = ['timeOpen', 'sentiment_mean', 'sentiment_neg_mean', 'relevance_score', 'events_count', 'negative_spike_flag']
        pd.DataFrame(columns=cols).to_csv(MINUTE_CSV, index=False)
        logger.info(f"Initialized {MINUTE_CSV}")

def aggregate_minute(events_csv: str, minute_csv: str):
    """Aggregate raw events into 1-minute bins for the dashboard/model."""
    if not os.path.exists(events_csv): return
    
    try:
        df = pd.read_csv(events_csv)
        df['time'] = pd.to_datetime(df['time'])
        
        # Filter relevant events (Broaden to 24h for stable summary)
        last_24h = datetime.now(timezone.utc) - timedelta(hours=24)
        # Ensure 'time' is localized or naive consistent. CSV read result might be naive.
        if df['time'].dt.tz is None:
             # Assume UTC if naive, as we write UTC
             df['time'] = df['time'].dt.tz_localize('UTC')
             
        df = df[df['time'] > last_24h]
        
        if df.empty: return

        # Resample
        df['timeOpen'] = df['time'].dt.floor('1min')
        
        agg = df.groupby('timeOpen').agg({
            'sentiment_compound': 'mean',
            'sentiment_neg': 'mean',
            'relevance_score': 'mean',
            'text': 'count'
        }).rename(columns={'sentiment_compound': 'sentiment_mean', 
                           'sentiment_neg': 'sentiment_neg_mean',
                           'text': 'events_count'})
        
        agg = agg.reset_index()
        
        # Negative Spike Flag
        threshold = SENTIMENT_CONFIG.get('thresholds', {}).get('negative_spike', -0.5)
        agg['negative_spike_flag'] = (agg['sentiment_mean'] < threshold).astype(int)
        
        # Append to minute csv (deduplicating by timeOpen)
        if os.path.exists(minute_csv):
            existing = pd.read_csv(minute_csv)
            existing['timeOpen'] = pd.to_datetime(existing['timeOpen'])
            if existing['timeOpen'].dt.tz is None:
                 existing['timeOpen'] = existing['timeOpen'].dt.tz_localize('UTC')
                 
            # Combine and update
            combined = pd.concat([existing, agg]).drop_duplicates(subset=['timeOpen'], keep='last')
            combined.sort_values('timeOpen', inplace=True)
            
            tmp_out = minute_csv + ".tmp"
            combined.to_csv(tmp_out, index=False)
            
            def safe_replace(tmp, target):
                max_retries = 10
                for i in range(max_retries):
                    try:
                        if os.path.exists(tmp):
                            os.replace(tmp, target)
                        return True
                    except PermissionError:
                        if i < max_retries - 1:
                            time.sleep(1.0)
                            continue
                        logger.warning(f"Sentiment: Could not update {target} due to file lock.")
                return False

            safe_replace(tmp_out, minute_csv)
        else:
            agg.to_csv(minute_csv, index=False)
            
    except Exception as e:
        logger.error(f"Aggregation Error: {e}")


# --- Main Loop ---

def stream_sources():
    sources = []
    
    # Configure Sources
    rss_conf = SENTIMENT_CONFIG.get('sources', {}).get('rss', {})
    if rss_conf.get('enabled'):
        sources.append(RSSSourceAdapter(rss_conf.get('feeds', [])))
        
    twitter_conf = SENTIMENT_CONFIG.get('sources', {}).get('twitter', {})
    if twitter_conf.get('enabled'):
        sources.append(TwitterSourceAdapter())

    while True:
        logger.info("Fetching from sources...")
        count = 0
        for source in sources:
            for event in source.fetch():
                if is_duplicate(event['text']):
                    continue
                
                scored_event = score_sentiment(event)
                if scored_event:
                    # Append raw event
                    append_event(scored_event, EVENTS_CSV)
                    count += 1
        
        if count > 0:
            logger.info(f"Processed {count} new sentiment events.")
        else:
            logger.info("No new relevant events.")
            
        # ⚠️ Always trigger aggregation ensure rolling window stays fresh
        aggregate_minute(EVENTS_CSV, MINUTE_CSV)
            
        time.sleep(SENTIMENT_CONFIG.get('interval_seconds', 60))

if __name__ == "__main__":
    logger.info("Starting Sentiment Ingest Service...")
    init_csvs()
    try:
        stream_sources()
    except KeyboardInterrupt:
        logger.info("Service stopped by user.")
    except Exception as e:
        logger.critical(f"Fatal Error: {e}", exc_info=True)
