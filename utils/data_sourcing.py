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

@st.cache_data(ttl=86400)
def varliklari_kesfet():
    with st.spinner("Piyasalar taranıyor ve potansiyel varlıklar keşfediliyor..."):
        aday_varliklar = []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        # --- BİST TARAMASI (SON GÜNCELLEME: Investing.com) ---
        try:
            # Birincil ve en güvenilir kaynak: Investing.com
            investing_url = "https://www.investing.com/indices/ise-100-components"
            response = requests.get(investing_url, headers=headers)
            tables = pd.read_html(response.content)
            bist_df = tables[0]
            # Investing.com'da semboller genellikle farklı bir sütunda olabilir, 'Symbol' veya benzeri bir isim arayalım
            symbol_col = next((col for col in bist_df.columns if 'symbol' in col.lower()), 'Symbol') # Varsayılan 'Symbol'
            bist100_tickerlar = [f"{ticker}.IS" for ticker in bist_df[symbol_col].str.strip()]
        except Exception:
            st.warning("BİST hisse listesi finans sitesinden çekilemedi. Acil durum listesi kullanılıyor.")
            bist100_tickerlar = [
                "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS",
                "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KOZAL.IS",
                "KRDMD.IS", "MGROS.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS",
                "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TTKOM.IS", "TUPRS.IS", "ULKER.IS",
                "VAKBN.IS", "YKBNK.IS", "SMRTG.IS", "HEKTS.IS", "ASTOR.IS", "KONTR.IS", "GESAN.IS"
            ]

        try:
            bist_skorlari = {ticker: (1/fa['pe_ratio'] + 1/fa['pb_ratio']) for ticker in bist100_tickerlar if (fa := get_fundamental_data(ticker)) and fa.get('pe_ratio')}
            en_iyi_bist = sorted(bist_skorlari, key=bist_skorlari.get, reverse=True)[:5]
            aday_varliklar.extend(en_iyi_bist)
        except Exception as e:
            st.warning(f"BİST temel analiz aşaması başarısız oldu: {e}")
            
        # --- NASDAQ TARAMASI (Wikipedia - Genellikle stabildir) ---
        try:
            nasdaq100_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response = requests.get(nasdaq100_url, headers=headers)
            nasdaq_tablolari = pd.read_html(response.content)
            nasdaq100_df = next((table for table in nasdaq_tablolari if 'Ticker' in table.columns and 'Company' in table.columns), None)
            if nasdaq100_df is None: raise ValueError("NASDAQ-100 tablosu bulunamadı.")
            
            nasdaq100_tickerlar = nasdaq100_df['Ticker'].str.strip().tolist()
            nasdaq_skorlari = {ticker: (1/fa['pe_ratio'] + 1/fa['pb_ratio']) for ticker in nasdaq100_tickerlar[:50] if (fa := get_fundamental_data(ticker)) and fa.get('pe_ratio')}
            en_iyi_nasdaq = sorted(nasdaq_skorlari, key=nasdaq_skorlari.get, reverse=True)[:5]
            aday_varliklar.extend(en_iyi_nasdaq)
        except Exception as e:
            st.warning(f"NASDAQ taraması başarısız oldu: {e}")

        # 3. Kripto ve Fonlar
        aday_varliklar.extend(["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"])
        aday_varliklar.extend(["SPY", "QQQ", "GLD", "USO", "IEMG"])

        son_liste = list(set(aday_varliklar))
        if not son_liste:
            st.error("Piyasa taraması sonucunda hiçbir uygun varlık bulunamadı.")
            return []
            
        st.success(f"{len(son_liste)} potansiyel varlık keşfedildi!")
        st.json(son_liste)
        return son_liste
