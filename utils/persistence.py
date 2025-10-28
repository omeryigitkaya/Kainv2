# utils/persistence.py
import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_gsheets import GSheetsConnection
import uuid
from datetime import datetime

def save_portfolio_to_gsheets(plan_tipi, optimal_agirliklar, yatirim_tutari):
    """
    Oluşturulan portföy önerisini ve içerdiği hisseleri Google Sheets'e kaydeder.
    """
    try:
        # 1. Google Sheets'e Bağlan
        # Bu bağlantı, Streamlit'in Secrets yönetimindeki "gsheets" anahtarını otomatik kullanır.
        conn = st.connection("gsheets", type=GSheetsConnection)

        # 2. Ana Portföy Bilgilerini Hazırla (portfolio_history tablosu için)
        portfolio_id = str(uuid.uuid4()) # Her portföy için benzersiz bir kimlik
        created_time = datetime.now()

        portfolio_history_data = pd.DataFrame([{
            "portfolio_id": portfolio_id,
            "created_timestamp": created_time,
            "plan_type": plan_tipi,
            "initial_investment": yatirim_tutari
        }])

        # 3. Portföydeki Hisselerin Detaylarını Hazırla (portfolio_holdings tablosu için)
        holdings_data = []
        tickers_list = list(optimal_agirliklar.keys())
        
        # yfinance'den tüm hisselerin anlık fiyatını tek seferde çekerek hız kazan
        current_prices = yf.download(tickers_list, period="1d", progress=False)['Close'].iloc[-1]

        for ticker, weight in optimal_agirliklar.items():
            purchase_price = current_prices[ticker]
            holdings_data.append({
                "holding_id": str(uuid.uuid4()),
                "portfolio_id": portfolio_id, # Ana tabloya bağlantı anahtarı
                "ticker": ticker,
                "weight": weight,
                "purchase_price": purchase_price # K/Z takibi için en kritik veri! [cite: 171]
            })
        
        portfolio_holdings_df = pd.DataFrame(holdings_data)

        # 4. Verileri Google Sheets'teki ilgili sayfalara ekle
        # Mevcut verilerin üzerine yazmaz, sonuna ekler.
        conn.update(worksheet="portfolio_history", data=portfolio_history_data)
        conn.update(worksheet="portfolio_holdings", data=portfolio_holdings_df)

        st.success(f"{plan_tipi} portföy önerisi başarıyla kaydedildi!")
        return True

    except Exception as e:
        st.error(f"Portföy kaydedilirken bir hata oluştu. Google Sheets ayarlarınızı kontrol edin. Hata: {e}")
        return False
