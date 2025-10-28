# utils/modeling.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import risk_models, EfficientFrontier, exceptions
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def cizim_yap_agirliklar(weights, ax=None):
    if ax is None: fig, ax = plt.subplots()
    labels = list(weights.keys()); sizes = list(weights.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90); ax.axis('equal')
    return ax.get_figure()

@st.cache_data(show_spinner=False)
def piyasa_rejimini_belirle():
    st.write("Piyasa rejimi analiz ediliyor...")
    rejim_gostergeleri = {"NASDAQ": "^IXIC", "BIST 100": "XU100.IS", "Altın": "GC=F", "Bitcoin": "BTC-USD", "ABD 10Y Faiz": "^TNX"}
    yonler = {"yukari": ["NASDAQ", "BIST 100", "Altın", "Bitcoin"], "asagi": ["ABD 10Y Faiz"]}
    toplam_puan = 0
    for isim, ticker in rejim_gostergeleri.items():
        try:
            veri = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
            if veri.empty: continue
            ma200 = veri['Close'].rolling(window=200).mean().iloc[-1]
            son_fiyat = veri['Close'].iloc[-1]
            puan = 0
            if isim in yonler['yukari'] and son_fiyat > ma200: puan = 1
            elif isim in yonler['yukari'] and son_fiyat < ma200: puan = -1
            elif isim in yonler['asagi'] and son_fiyat < ma200: puan = 1
            elif isim in yonler['asagi'] and son_fiyat > ma200: puan = -1
            toplam_puan += puan
        except Exception: continue
    if toplam_puan >= 3: return "GÜÇLÜ POZİTİF (BOĞA 🐂🐂)"
    elif toplam_puan >= 1: return "TEMKİNLİ POZİTİF (BOĞA 🐂)"
    else: return "TEMKİNLİ NEGATİF (AYI 🐻)"

@st.cache_data
def veri_cek_ve_dogrula(tickers, start, end):
    data = {t: yf.download(t, start=start, end=end, progress=False, auto_adjust=True)['Close'] for t in tickers}
    valid_data = {t: v for t, v in data.items() if not v.empty and len(v.dropna()) > 260}
    if not valid_data: return pd.DataFrame()
    return pd.concat(valid_data, axis=1).ffill().dropna()

@st.cache_data
def sinyal_uret_ensemble_lstm(fiyat_verisi):
    predictions = []
    for look_back in [12, 26]:
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(fiyat_verisi.values.reshape(-1, 1))
            X_train, y_train = [scaled_data[i-look_back:i, 0] for i in range(look_back, len(scaled_data))], [scaled_data[i, 0] for i in range(look_back, len(scaled_data))]
            if not X_train: continue
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            model = Sequential([LSTM(50), Dropout(0.2), Dense(1)])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
            X_test = np.array([scaled_data[-look_back:].flatten()])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_price = scaler.inverse_transform(model.predict(X_test, verbose=0))
            predictions.append(predicted_price[0][0])
        except Exception: continue
    last_known_price = fiyat_verisi.iloc[-1]
    if not predictions: return {"tahmin_yuzde": 0.0, "son_fiyat": last_known_price, "hedef_fiyat": last_known_price}
    ortalama_hedef_fiyat = np.mean(predictions)
    percentage_change = (ortalama_hedef_fiyat - last_known_price) / last_known_price
    return {"tahmin_yuzde": percentage_change, "son_fiyat": last_known_price, "hedef_fiyat": ortalama_hedef_fiyat}

@st.cache_data
def sinyal_uret_ceyrekli_momentum(fiyat_serisi):
    son_fiyat = fiyat_serisi.iloc[-1]
    # Son 6 aylık (yaklaşık 126 işlem günü) performansı hesapla
    fiyat_6ay_once = fiyat_serisi.iloc[-126] if len(fiyat_serisi) > 126 else fiyat_serisi.iloc[0]
    ceyrekli_getiri = (son_fiyat / fiyat_6ay_once) - 1
    hedef_fiyat = son_fiyat * (1 + ceyrekli_getiri)
    return {"tahmin_yuzde": ceyrekli_getiri, "son_fiyat": son_fiyat, "hedef_fiyat": hedef_fiyat}

@st.cache_data
def calculate_multi_factor_score(faktör_verileri, agirliklar):
    df = pd.DataFrame(faktör_verileri).T.astype(float)
    df.fillna(df.mean(), inplace=True)
    df_zscore = df.copy()
    for col in df.columns:
        col_std = df[col].std()
        if col_std > 0:
            df_zscore[col] = (df[col] - df[col].mean()) / col_std
        else:
            df_zscore[col] = 0
    final_scores = (df_zscore * pd.Series(agirliklar)).sum(axis=1)
    return final_scores.fillna(0)

@st.cache_data
def portfoyu_optimize_et(nihai_skorlar, fiyat_verisi, piyasa_rejimi):
    if nihai_skorlar.empty or nihai_skorlar.std() == 0:
        st.warning("Sinyaller anlamsız veya tekdüze. Varlıklar eşit olarak dağıtılacak.")
        num_assets = len(fiyat_verisi.columns)
        return {ticker: 1/num_assets for ticker in fiyat_verisi.columns}
    mu = nihai_skorlar
    S = risk_models.CovarianceShrinkage(fiyat_verisi[mu.index]).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    try:
        weights = ef.max_quadratic_utility()
    except (exceptions.InstantiationError, ValueError):
        try:
            weights = ef.min_volatility()
        except (exceptions.InstantiationError, ValueError):
            st.warning("Optimizasyon başarısız oldu. Varlıklar eşit olarak dağıtılacak.")
            num_assets = len(fiyat_verisi.columns)
            return {ticker: 1/num_assets for ticker in fiyat_verisi.columns}
    return ef.clean_weights()
