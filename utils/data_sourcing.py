# utils/data_sourcing.py

import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd

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
            url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={ticker.split('.')[0]}"
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

# --- YENİ VARLIK KEŞİF MOTORU ---
@st.cache_data(ttl=86400) # Günde bir kez piyasayı tara
def varliklari_kesfet(max_varlik_sayisi=20):
    """
    Farklı piyasalardan temel kriterlere göre potansiyel varlıkları keşfeder.
    """
    with st.spinner("Piyasalar taranıyor ve potansiyel varlıklar keşfediliyor..."):
        aday_varliklar = []

        # 1. BİST Taraması (BIST100 içinden en ucuz 5 hisse)
        try:
            bist100_url = "https://tr.wikipedia.org/wiki/BIST_100"
            bist_tablolari = pd.read_html(bist100_url)
            bist100_df = bist_tablolari[1] # Genellikle ikinci tablo
            bist100_tickerlar = [f"{ticker}.IS" for ticker in bist100_df['Kod'].str.strip()]
            
            bist_skorlari = {}
            for ticker in bist100_tickerlar[:30]: # API limitlerini zorlamamak için ilk 30'u alalım
                fa = get_fundamental_data(ticker)
                if fa and fa.get('pe_ratio') and fa.get('pb_ratio'):
                    bist_skorlari[ticker] = (1/fa['pe_ratio']) + (1/fa['pb_ratio'])
            
            # En yüksek skora sahip (en ucuz) 5 hisseyi seç
            en_iyi_bist = sorted(bist_skorlari, key=bist_skorlari.get, reverse=True)[:5]
            aday_varliklar.extend(en_iyi_bist)
        except Exception as e:
            st.warning(f"BİST taraması başarısız oldu: {e}")

        # 2. NASDAQ Taraması (NASDAQ100 içinden en ucuz 5 hisse)
        try:
            nasdaq100_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            nasdaq_tablolari = pd.read_html(nasdaq100_url)
            nasdaq100_df = nasdaq_tablolari[4] # Genellikle beşinci tablo
            nasdaq100_tickerlar = nasdaq100_df['Ticker'].str.strip().tolist()

            nasdaq_skorlari = {}
            for ticker in nasdaq100_tickerlar[:30]: # API limitleri için
                fa = get_fundamental_data(ticker)
                if fa and fa.get('pe_ratio') and fa.get('pb_ratio'):
                    nasdaq_skorlari[ticker] = (1/fa['pe_ratio']) + (1/fa['pb_ratio'])
            
            en_iyi_nasdaq = sorted(nasdaq_skorlari, key=nasdaq_skorlari.get, reverse=True)[:5]
            aday_varliklar.extend(en_iyi_nasdaq)
        except Exception as e:
            st.warning(f"NASDAQ taraması başarısız oldu: {e}")

        # 3. Kripto Varlıkları (En büyük 5)
        kriptolar = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]
        aday_varliklar.extend(kriptolar)

        # 4. Fonlar/ETF'ler (Çeşitlendirme için)
        etfler = ["SPY", "QQQ", "GLD", "USO", "IEMG"] # S&P500, NASDAQ, Altın, Petrol, Gelişen Piyasalar
        aday_varliklar.extend(etfler)

        # Tekrarlananları temizle ve sonucu döndür
        son_liste = list(set(aday_varliklar))
        st.success(f"{len(son_liste)} potansiyel varlık keşfedildi!")
        st.json(son_liste)
        return son_liste
