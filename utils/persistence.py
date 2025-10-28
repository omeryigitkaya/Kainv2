# utils/persistence.py
import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_gsheets import GSheetsConnection
import uuid
from datetime import datetime

# =============================================================================
# BÖLÜM 3: VERİ KAYDETME (Önceki Adımdan)
# =============================================================================

def save_portfolio_to_gsheets(plan_tipi, optimal_agirliklar, yatirim_tutari):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        portfolio_id = str(uuid.uuid4())
        created_time = datetime.now()

        portfolio_history_data = pd.DataFrame([{
            "portfolio_id": portfolio_id, "created_timestamp": created_time,
            "plan_type": plan_tipi, "initial_investment": yatirim_tutari
        }])

        holdings_data = []
        tickers_list = list(optimal_agirliklar.keys())
        current_prices = yf.download(tickers_list, period="1d", progress=False)['Close'].iloc[-1]

        for ticker, weight in optimal_agirliklar.items():
            purchase_price = current_prices.get(ticker)
            if purchase_price and not pd.isna(purchase_price):
                 holdings_data.append({
                    "holding_id": str(uuid.uuid4()), "portfolio_id": portfolio_id,
                    "ticker": ticker, "weight": weight, "purchase_price": purchase_price
                })
        
        if not holdings_data:
            st.warning("Hisselerin anlık fiyatları alınamadığı için portföy detayları kaydedilemedi.")
            return False

        portfolio_holdings_df = pd.DataFrame(holdings_data)
        conn.update(worksheet="portfolio_history", data=portfolio_history_data)
        conn.update(worksheet="portfolio_holdings", data=portfolio_holdings_df)
        st.success(f"{plan_tipi} portföy önerisi başarıyla kaydedildi!")
        return True
    except Exception as e:
        st.error(f"Portföy kaydedilirken bir hata oluştu. Hata: {e}")
        return False

# =============================================================================
# BÖLÜM 4: VERİ OKUMA VE K/Z HESAPLAMA (YENİ)
# =============================================================================

@st.cache_data(ttl=900) # 15 dakika önbellekle
def load_all_portfolios_from_gsheets():
    """
    Google Sheets'ten kaydedilmiş tüm portföylerin ana bilgilerini çeker.
    """
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # created_timestamp sütununu tarih formatına çevirerek oku
        portfolios = conn.read(worksheet="portfolio_history", usecols=list(range(4)), header=0)
        portfolios['created_timestamp'] = pd.to_datetime(portfolios['created_timestamp'])
        return portfolios.dropna(subset=['portfolio_id'])
    except Exception:
        st.warning("Geçmiş portföy verileri okunamadı. Google Sheets bağlantınızı kontrol edin veya henüz hiç portföy kaydetmediniz.")
        return pd.DataFrame()

@st.cache_data(ttl=900) # 15 dakika önbellekle
def calculate_pl(selected_portfolio_id):
    """
    Seçilen bir portföyün anlık Kar/Zarar durumunu hesaplar.
    """
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # 1. Portföydeki hisseleri ve alım fiyatlarını çek
        all_holdings = conn.read(worksheet="portfolio_holdings", usecols=list(range(5)), header=0)
        portfolio_holdings = all_holdings[all_holdings['portfolio_id'] == selected_portfolio_id]

        if portfolio_holdings.empty:
            return None

        # 2. Hisselerin anlık fiyatlarını çek
        tickers_list = portfolio_holdings['ticker'].tolist()
        current_prices = yf.download(tickers_list, period="1d", progress=False)['Close'].iloc[-1]

        # 3. K/Z Hesapla
        pl_data = []
        total_portfolio_return = 0
        for index, row in portfolio_holdings.iterrows():
            ticker = row['ticker']
            weight = float(row['weight'])
            purchase_price = float(row['purchase_price'])
            current_price = current_prices.get(ticker)
            
            if current_price and not pd.isna(current_price):
                stock_return = (current_price / purchase_price) - 1
                total_portfolio_return += stock_return * weight
                pl_data.append({
                    "Varlık": ticker,
                    "Alım Fiyatı ($)": purchase_price,
                    "Anlık Fiyat ($)": current_price,
                    "Ağırlık": weight,
                    "Bireysel Getiri": stock_return
                })
        
        pl_df = pd.DataFrame(pl_data)
        return {"total_return": total_portfolio_return, "details_df": pl_df, "holdings": portfolio_holdings}

    except Exception as e:
        st.error(f"K/Z hesaplanırken bir hata oluştu: {e}")
        return None
