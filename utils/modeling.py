# utils/modeling.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import BlackLittermanModel, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from scipy.stats import zscore

# =============================================================================
# BÖLÜM 0: MEVCUT YARDIMCI FONKSİYONLAR (DEĞİŞİKLİK YOK)
# =============================================================================

def cizim_yap_agirliklar(weights, ax=None):
    if ax is None: fig, ax = plt.subplots()
    labels = list(weights.keys()); sizes = list(weights.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90); ax.axis('equal')
    return ax.get_figure()

@st.cache_data(show_spinner=False)
def piyasa_rejimini_belirle():
    st.write("Piyasa rejimi analiz ediliyor...")
    rejim_gostergeleri = {"NASDAQ": {"ticker": "^IXIC", "yon": "yukari"},"BIST 100": {"ticker": "XU100.IS", "yon": "yukari"},"Altın": {"ticker": "GC=F", "yon": "yukari"},"Bitcoin": {"ticker": "BTC-USD", "yon": "yukari"},"ABD 10Y Faiz": {"ticker": "^TNX", "yon": "asagi"}}
    toplam_puan = 0
    for isim, info in rejim_gostergeleri.items():
        try:
            veri = yf.download(info['ticker'], period="2y", progress=False, auto_adjust=True)
            if veri is None or veri.empty: continue
            veri['MA200'] = veri['Close'].rolling(window=200).mean()
            son_fiyat = veri['Close'].iloc[-1]; son_ma200 = veri['MA200'].iloc[-1]
            if not np.isfinite(son_fiyat) or not np.isfinite(son_ma200): continue
            puan = 1 if (info['yon'] == 'yukari' and son_fiyat > son_ma200) or \
                         (info['yon'] == 'asagi' and son_fiyat < son_ma200) else -1
            toplam_puan += puan
        except Exception: continue

    if toplam_puan >= 3: return "GÜÇLÜ POZİTİF (BOĞA 🐂🐂)"
    elif toplam_puan >= 1: return "TEMKİNLİ POZİTİF (BOĞA 🐂)"
    else: return "TEMKİNLİ NEGATİF (AYI 🐻)"

@st.cache_data
def veri_cek_ve_dogrula(tickers, start, end):
    gecerli_datalar = {t: yf.download(t, start=start, end=end, progress=False, auto_adjust=True)['Close'] for t in tickers}
    gecerli_datalar = {t: v for t, v in gecerli_datalar.items() if not v.empty and len(v.dropna()) > 260} # En az 1 yıl veri olsun

    if not gecerli_datalar:
        st.warning("Hiçbir varlık için yeterli veri bulunamadı.")
        return pd.DataFrame()

    close_prices_df = pd.concat(gecerli_datalar, axis=1).ffill().dropna()
    st.info(f"Analize devam edilecek geçerli varlıklar: {list(close_prices_df.columns)}")
    return close_prices_df

@st.cache_data
def sinyal_uret_ensemble_lstm(fiyat_verisi, look_back_periods=[12, 26, 52]):
    # Bu fonksiyon (Teknik/Momentum Faktörü) rapordaki gibi yeniden konumlandırıldı.
    # Çıktısı artık tek başına bir karar değil, hibrit modelin bir girdisi olacak.
    predictions = []
    for look_back in look_back_periods:
        try:
            scaler = MinMaxScaler(feature_range=(0, 1)); scaled_data = scaler.fit_transform(fiyat_verisi.values.reshape(-1, 1)); X_train, y_train = [], []
            for i in range(look_back, len(scaled_data)):
                X_train.append(scaled_data[i-look_back:i, 0]); y_train.append(scaled_data[i, 0])
            if not X_train: continue
            X_train, y_train = np.array(X_train), np.array(y_train); X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            model = Sequential([LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)), Dropout(0.2), LSTM(units=50, return_sequences=False), Dropout(0.2), Dense(units=1)])
            model.compile(optimizer='adam', loss='mean_squared_error'); model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
            last_look_back_weeks = scaled_data[-look_back:]; X_test = np.array([last_look_back_weeks.flatten()]); X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_scaled = model.predict(X_test, verbose=0); predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
            predictions.append(predicted_price)
        except Exception: continue
    last_known_price = fiyat_verisi.iloc[-1]
    if not predictions: return 0.0
    ortalama_hedef_fiyat = np.mean(predictions)
    return ((ortalama_hedef_fiyat - last_known_price) / last_known_price)

# =============================================================================
# BÖLÜM 2: YENİ ÇOK FAKTÖRLÜ HİBRİT MODEL MANTIĞI
# =============================================================================

@st.cache_data
def calculate_multi_factor_score(faktör_verileri, agirliklar):
    """
    Her hisse için verilen faktörleri (Değer, Duyarlılık, Teknik)
    Z-skor ile normalize eder ve verilen ağırlıklara göre nihai bir skor hesaplar.
    """
    df = pd.DataFrame(faktör_verileri).T
    
    # Boş değerleri o faktörün ortalaması ile doldur
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

    # Z-skor hesapla: Tüm faktörleri karşılaştırılabilir hale getir
    df_zscore = df.apply(zscore)

    # Ağırlıklı nihai skoru hesapla
    nihai_skor = (df_zscore * pd.Series(agirliklar)).sum(axis=1)
    
    return nihai_skor

@st.cache_data
def portfoyu_optimize_et(nihai_skorlar, fiyat_verisi, piyasa_rejimi):
    """
    Artık sadece LSTM sinyalini değil, çok faktörlü nihai skoru kullanarak
    portföyü optimize eder.
    """
    if nihai_skorlar.empty: return {}
    
    fiyat_verisi = fiyat_verisi[nihai_skorlar.index]
    
    # Strateji belirle
    if "POZİTİF" in piyasa_rejimi:
        agirlik_limiti, hedef = 0.60, "max_sharpe"
    else:
        agirlik_limiti, hedef = max(0.35, 1/len(fiyat_verisi.columns)), "min_volatility"

    S = risk_models.sample_cov(fiyat_verisi)
    
    # Nihai skorları beklenen getiri olarak kullan
    # PyPortfolioOpt'un beklediği formata getiriyoruz
    mu = nihai_skorlar
    
    ef = EfficientFrontier(mu, S, weight_bounds=(0, agirlik_limiti))
    try:
        weights = ef.max_sharpe() if hedef == "max_sharpe" else ef.min_volatility()
    except (ValueError, OptimizationError):
        try: 
            weights = ef.min_volatility()
        except (ValueError, OptimizationError): 
            # Hiçbir optimizasyon çalışmazsa eşit ağırlık ver
            num_assets = len(fiyat_verisi.columns)
            weights = {ticker: 1/num_assets for ticker in fiyat_verisi.columns}

    # Küçük ağırlıkları temizle
    cleaned_weights = {k: v for k, v in weights.items() if v > 0.001}
    return cleaned_weights
