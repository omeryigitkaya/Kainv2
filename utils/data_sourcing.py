# utils/data_sourcing.py

import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
# Gelişmiş ve sadece Türkçe bilen model yerine, evrensel NLTK VADER modelini kullanıyoruz.
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# NLTK VADER modelinin çalışması için gerekli olan bir kerelik indirme işlemi
# Streamlit Cloud'da her zaman çalışacaktır.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# =============================================================================
# BÖLÜM 1.1: TEMEL ANALİZ VERİLERİ (FA)
# =============================================================================
@st.cache_data(ttl=86400)
def get_fundamental_data(ticker):
    # Amerikan hisseleri için yfinance genellikle yeterlidir.
    try:
        stock_info = yf.Ticker(ticker).info
        pe_ratio = stock_info.get('trailingPE')
        pb_ratio = stock_info.get('priceToBook')
        if pe_ratio is not None and pb_ratio is not None:
            return {'pe_ratio': pe_ratio, 'pb_ratio': pb_ratio}
    except Exception:
        pass

    # Sadece BİST hisseleri için yedek strateji (Amerikan hisselerinde hata verecek ve atlanacak)
    if '.IS' in ticker:
        try:
            url = f"https.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={ticker.split('.')[0]}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'lxml')
            pe_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutF_K').text.strip()
            pb_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutPD_DD').text.strip()
            return {'pe_ratio': float(pe_value.replace(',', '.')), 'pb_ratio': float(pb_value.replace(',', '.'))}
        except Exception:
            pass # Hata olursa görmezden gel, aşağıda None dönecek.

    return {'pe_ratio': None, 'pb_ratio': None}

# =============================================================================
# BÖLÜM 1.2: DUYARLILIK ANALİZİ VERİLERİ (SA) - EVRENSEL MODEL
# =============================================================================
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_data(ttl=3600)
def get_sentiment_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.0
        
        headlines = [article['title'] for article in news if 'title' in article and article['title']]
        if not headlines: return 0.0

        sia = load_sentiment_analyzer()
        
        # Her başlığın 'compound' skorunu al ve ortalamasını hesapla
        scores = [sia.polarity_scores(headline)['compound'] for headline in headlines]
        
        return sum(scores) / len(scores) if scores else 0.0
    except Exception:
        st.warning(f"{ticker} için duyarlılık analizi yapılamadı.")
        return 0.0
