# utils/persistence.py

import streamlit as st
import pandas as pd
import yfinance as yf
import uuid
from datetime import datetime

# st.connection yerleşik olduğu için harici kütüphaneye gerek kalmadı

def save_portfolio_to_gsheets(plan_tipi, optimal_agirliklar, yatirim_tutari):
    try:
        # Streamlit'in yerleşik gsheets bağlantısını kullanıyoruz
        conn = st.connection("gsheets", type="sheets")

        portfolio_id = str(uuid.uuid4())
        created_time = datetime.now()

        # Yeni satırları DataFrame olarak hazırlıyoruz
        history_df = pd.DataFrame([{"portfolio_id": portfolio_id, "created_timestamp": created_time, "plan_type": plan_tipi, "initial_investment": yatirim_tutari}])
        
        tickers_list = list(optimal_agirliklar.keys())
        prices_df = yf.download(tickers_list, period="1d", progress=False, auto_adjust=True)
        current_prices = prices_df['Close'].iloc[-1] if isinstance(prices_df['Close'], pd.Series) else prices_df['Close']
        
        holdings_data = []
        for ticker, weight in optimal_agirliklar.items():
            price = current_prices[ticker] if len(tickers_list) > 1 else current_prices
            if pd.notna(price):
                holdings_data.append({"holding_id": str(uuid.uuid4()), "portfolio_id": portfolio_id, "ticker": ticker, "weight": weight, "purchase_price": price})
        
        if not holdings_data:
            st.warning("Hisse fiyatları alınamadığı için portföy kaydedilemedi."); return False
        
        holdings_df = pd.DataFrame(holdings_data)

        # Mevcut veriyi okuyup yenisini ekliyoruz
        existing_history = conn.read(worksheet="portfolio_history", usecols=list(range(4)))
        updated_history = pd.concat([existing_history, history_df], ignore_index=True)
        conn.write(worksheet="portfolio_history", data=updated_history)

        existing_holdings = conn.read(worksheet="portfolio_holdings", usecols=list(range(5)))
        updated_holdings = pd.concat([existing_holdings, holdings_df], ignore_index=True)
        conn.write(worksheet="portfolio_holdings", data=updated_holdings)

        st.success(f"{plan_tipi} portföy önerisi başarıyla kaydedildi!")
        return True
    except Exception as e:
        st.error(f"Portföy kaydedilemedi. Secrets veya Google Sheets API ayarlarınızı kontrol edin. Hata: {e}"); return False

@st.cache_data(ttl=900)
def load_all_portfolios_from_gsheets():
    try:
        conn = st.connection("gsheets", type="sheets")
        portfolios = conn.read(worksheet="portfolio_history", usecols=list(range(4)))
        portfolios['created_timestamp'] = pd.to_datetime(portfolios['created_timestamp'])
        return portfolios.dropna(subset=['portfolio_id'])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def calculate_pl(selected_portfolio_id):
    try:
        conn = st.connection("gsheets", type="sheets")
        all_holdings = conn.read(worksheet="portfolio_holdings", usecols=list(range(5)))
        portfolio_holdings = all_holdings[all_holdings['portfolio_id'] == selected_portfolio_id]
        if portfolio_holdings.empty: return None

        tickers_list = portfolio_holdings['ticker'].tolist()
        prices_df = yf.download(tickers_list, period="1d", progress=False, auto_adjust=True)
        current_prices = prices_df['Close'].iloc[-1] if isinstance(prices_df['Close'], pd.Series) else prices_df['Close']

        pl_data = []
        total_return = 0
        for _, row in portfolio_holdings.iterrows():
            ticker, weight, purchase_price = row['ticker'], float(row['weight']), float(row['purchase_price'])
            current_price = current_prices[ticker] if len(tickers_list) > 1 else current_prices
            if pd.notna(current_price):
                stock_return = (current_price / purchase_price) - 1
                total_return += stock_return * weight
                pl_data.append({"Varlık": ticker, "Alım Fiyatı ($)": purchase_price, "Anlık Fiyat ($)": current_price, "Ağırlık": weight, "Bireysel Getiri": stock_return})
        
        return {"total_return": total_return, "details_df": pd.DataFrame(pl_data), "holdings": portfolio_holdings}
    except Exception as e:
        st.error(f"K/Z hesaplanırken hata: {e}"); return None
