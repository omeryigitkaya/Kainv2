# utils/data_sourcing.py
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Raporun önerdiği gibi, bu fonksiyon 24 saat boyunca sonuçları hafızasında tutacak.
# Bu sayede her seferinde yeniden veri çekerek programı yavaşlatmayacak.
@st.cache_data(ttl=86400)
def get_fundamental_data(ticker):
    """
    Bir hisse senedi için Temel Analiz verilerini (F/K ve PD/DD) çeker.
    Önce yfinance kütüphanesini dener, başarısız olursa web scraping (yedek strateji) kullanır.
    """
    try:
        # 1. Strateji: yfinance kütüphanesini dene (Hızlı ve Öncelikli)
        stock_info = yf.Ticker(ticker).info
        pe_ratio = stock_info.get('trailingPE')
        pb_ratio = stock_info.get('priceToBook')

        # Eğer yfinance veri bulamazsa (None dönerse), yedek stratejiye geç
        if pe_ratio is not None and pb_ratio is not None:
            return {'pe_ratio': pe_ratio, 'pb_ratio': pb_ratio}

    except Exception:
        # yfinance'de herhangi bir hata olursa sessizce görmezden gel ve yedek stratejiye geç
        pass

    # 2. Strateji: Web Scraping (Yedek - İş Yatırım'dan veri çekme)
    try:
        url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={ticker.split('.')[0]}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # F/K ve PD/DD değerlerini tablodan bul ve al
        pe_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutF_K').text.strip()
        pb_value = soup.find(id='ctl00_ctl58_g_8628006e_4f1d_4296_8823_9997c48f8859_ctl00_MevcutPD_DD').text.strip()

        # Metinleri sayısal değere çevir
        pe_ratio_scraped = float(pe_value.replace(',', '.'))
        pb_ratio_scraped = float(pb_value.replace(',', '.'))

        return {'pe_ratio': pe_ratio_scraped, 'pb_ratio': pb_ratio_scraped}

    except Exception as e:
        st.warning(f"{ticker} için temel veriler yfinance veya web scraping ile alınamadı. Hata: {e}")
        return {'pe_ratio': None, 'pb_ratio': None}
