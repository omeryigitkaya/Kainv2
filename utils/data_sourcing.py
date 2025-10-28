# utils/data_sourcing.py
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from financelib.stocks import Stocks
from transformers import pipeline
import numpy as np

# =============================================================================
# BÖLÜM 1.1: TEMEL ANALİZ VERİLERİ (FA)
# =============================================================================

@st.cache_data(ttl=86400) # 24 saat önbellekle [cite: 36]
def get_fundamental_data(ticker):
    """
    Bir hisse senedi için Temel Analiz verilerini (F/K ve PD/DD) çeker.
    Önce yfinance kütüphanesini dener, başarısız olursa web scraping kullanır.
    """
    try:
        # 1. Strateji: yfinance kütüphanesi [cite: 26]
        stock_info = yf.Ticker(ticker).info
        pe_ratio = stock_info.get('trailingPE')
        pb_ratio = stock_info.get('priceToBook')

        if pe_ratio is not None and pb_ratio is not None:
            return {'pe_ratio': pe_ratio, 'pb_ratio': pb_ratio}
    except Exception:
        pass

    # 2. Strateji: Web Scraping (Yedek) [cite: 30]
    try:
        url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={ticker.split('.')[0]}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        pe_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutF_K').text.strip()
        pb_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutPD_DD').text.strip()

        pe_ratio_scraped = float(pe_value.replace(',', '.'))
        pb_ratio_scraped = float(pb_value.replace(',', '.'))

        return {'pe_ratio': pe_ratio_scraped, 'pb_ratio': pb_ratio_scraped}
    except Exception as e:
        st.warning(f"{ticker} için temel veriler alınamadı. Hata: {e}")
        return {'pe_ratio': None, 'pb_ratio': None}

# =============================================================================
# BÖLÜM 1.2: DUYARLILIK ANALİZİ VERİLERİ (SA)
# =============================================================================

# Bu fonksiyon, ağır yapay zeka modelini sadece bir kez yükleyip hafızada tutar. [cite: 72]
@st.cache_resource
def load_sentiment_model():
    """
    Hugging Face'den önceden eğitilmiş bir Türkçe duyarlılık analiz modelini yükler.
    """
    model = pipeline(
        "sentiment-analysis",
        model="savasy/bert-base-turkish-sentiment-cased",
        max_length=512,
        truncation=True
    )
    return model

# Bu fonksiyon, her bir hisse için haberleri çeker ve 1 saat boyunca sonucu hafızada tutar. [cite: 74]
@st.cache_data(ttl=3600)
def get_sentiment_score(ticker):
    """
    Bir hisse senedi için financelib kullanarak haber başlıklarını çeker,
    BERT modeli ile analiz eder ve -1 (negatif) ile +1 (pozitif) arasında
    tek bir duyarlılık skoruna dönüştürür. [cite: 70]
    """
    try:
        # 1. Haber başlıklarını çek [cite: 67]
        stock_name = ticker.split('.')[0]
        news = Stocks(stock_name, "turkey").get_news()
        if not news:
            return 0.0

        # Sadece başlıkları al
        headlines = [n['title'] for n in news if 'title' in n]
        if not headlines:
            return 0.0

        # 2. Modeli yükle ve analiz yap [cite: 69]
        model = load_sentiment_model()
        results = model(headlines)

        # 3. Sonuçları tek bir skora dönüştür
        # Model 'positive' veya 'LABEL_1' gibi etiketler dönebilir, ikisini de kontrol edelim
        # 'positive' ise +1, 'negative' ise -1 puan verelim.
        score = 0
        for res in results:
            if res['label'].lower() in ['positive', '4 stars', '5 stars']:
                score += res['score']
            elif res['label'].lower() in ['negative', '1 star', '2 stars']:
                score -= res['score']
        
        # Skoru -1 ile +1 arasına normalize etmek için ortalamasını alalım
        return score / len(results) if results else 0.0

    except Exception as e:
        st.warning(f"{ticker} için duyarlılık analizi yapılamadı. Hata: {e}")
        return 0.0
