# utils/data_sourcing.py

import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
# financelib kaldırıldı, onun yerine yfinance kullanılacak
from transformers import pipeline, logging as hf_logging

# transformers kütüphanesinin çok fazla uyarı mesajı basmasını engelle
hf_logging.set_verbosity_error()

# =============================================================================
# BÖLÜM 1.1: TEMEL ANALİZ VERİLERİ (FA)
# =============================================================================
@st.cache_data(ttl=86400) # 24 saat önbellekle
def get_fundamental_data(ticker):
    try:
        stock_info = yf.Ticker(ticker).info
        pe_ratio = stock_info.get('trailingPE')
        pb_ratio = stock_info.get('priceToBook')
        if pe_ratio is not None and pb_ratio is not None:
            return {'pe_ratio': pe_ratio, 'pb_ratio': pb_ratio}
    except Exception:
        pass

    try:
        url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={ticker.split('.')[0]}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        pe_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutF_K').text.strip()
        pb_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutPD_DD').text.strip()
        return {'pe_ratio': float(pe_value.replace(',', '.')), 'pb_ratio': float(pb_value.replace(',', '.'))}
    except Exception:
        st.warning(f"{ticker} için temel veriler alınamadı.")
        return {'pe_ratio': None, 'pb_ratio': None}

# =============================================================================
# BÖLÜM 1.2: DUYARLILIK ANALİZİ VERİLERİ (SA)
# =============================================================================
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased", max_length=512, truncation=True)

@st.cache_data(ttl=3600) # 1 saat önbellekle
def get_sentiment_score(ticker):
    try:
        # Haber kaynağı olarak financelib yerine yfinance kullanıyoruz.
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.0
        
        # yfinance'den gelen haberlerin başlıklarını alıyoruz
        headlines = [article['title'] for article in news if 'title' in article and article['title']]
        if not headlines: return 0.0

        model = load_sentiment_model()
        results = model(headlines)
        
        score = 0
        for res in results:
            if res['label'].lower() in ['positive', '4 stars', '5 stars']:
                score += res['score']
            elif res['label'].lower() in ['negative', '1 star', '2 stars']:
                score -= res['score']
        return score / len(results) if results else 0.0
    except Exception:
        st.warning(f"{ticker} için duyarlılık analizi yapılamadı.")
        return 0.0
