# utils/data_sourcing.py

import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

@st.cache_data(ttl=86400)
def get_fundamental_data(ticker):
    try:
        stock_info = yf.Ticker(ticker).info
        pe_ratio = stock_info.get('trailingPE')
        pb_ratio = stock_info.get('priceToBook')
        if pe_ratio is not None and pb_ratio is not None and pe_ratio > 0 and pb_ratio > 0:
            return {'pe_ratio': pe_ratio, 'pb_ratio': pb_ratio}
    except Exception:
        pass

    if '.IS' in ticker:
        try:
            url = f"https.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={ticker.split('.')[0]}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'lxml')
            pe_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutF_K').text.strip()
            pb_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutPD_DD').text.strip()
            pe_ratio_scraped = float(pe_value.replace(',', '.'))
            pb_ratio_scraped = float(pb_value.replace(',', '.'))
            if pe_ratio_scraped > 0 and pb_ratio_scraped > 0:
                return {'pe_ratio': pe_ratio_scraped, 'pb_ratio': pb_ratio_scraped}
        except Exception:
            pass
            
    return {'pe_ratio': None, 'pb_ratio': None}

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_data(ttl=3600)
def get_sentiment_score(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news: return 0.0
        headlines = [article['title'] for article in news if 'title' in article]
        if not headlines: return 0.0
        sia = load_sentiment_analyzer()
        scores = [sia.polarity_scores(headline)['compound'] for headline in headlines]
        return sum(scores) / len(scores) if scores else 0.0
    except Exception:
        return 0.0
