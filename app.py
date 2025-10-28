# app.py

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.express as px

from utils.modeling import (piyasa_rejimini_belirle, veri_cek_ve_dogrula, sinyal_uret_ensemble_lstm,
                            sinyal_uret_yillik_momentum, calculate_multi_factor_score, 
                            portfoyu_optimize_et, cizim_yap_agirliklar)
from utils.data_sourcing import get_fundamental_data, get_sentiment_score
from utils.persistence import (save_portfolio_to_gsheets, load_all_portfolios_from_gsheets, calculate_pl)

st.set_page_config(layout="wide", page_title="Kainvest 2.0")

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True; del st.session_state["password"]
        else: st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password"); return False
    elif not st.session_state["password_correct"]:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password"); st.error("😕 Şifre yanlış."); return False
    return True

def get_tickers_from_github(user, repo, path):
    url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{path}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [t.strip() for t in response.text.strip().splitlines() if t.strip()]
    except requests.RequestException as e:
        st.error(f"Varlık listesi çekilemedi: {e}"); return None

def run_analysis(plan_tipi, agirliklar, tickers, yatirim_tutari):
    with st.spinner(f"{plan_tipi} portföy analiz ediliyor..."):
        rejim = piyasa_rejimini_belirle()
        st.subheader(f"Piyasa Rejimi: {rejim}")
        
        fiyatlar = veri_cek_ve_dogrula(tickers, "2020-01-01", pd.to_datetime("today").strftime('%Y-%m-%d'))
        if fiyatlar.empty: st.error("Analiz için yeterli veri bulunamadı."); return

        faktörler = {}
        sinyal_detaylari = {} 

        progress_bar = st.progress(0, text="Sinyaller üretiliyor...")
        for i, ticker in enumerate(fiyatlar.columns):
            if plan_tipi == "Haftalık":
                sinyal_data = sinyal_uret_ensemble_lstm(fiyatlar[ticker])
                teknik_skor = sinyal_data["tahmin_yuzde"]
            else: # Yıllık Plan
                sinyal_data = sinyal_uret_yillik_momentum(fiyatlar[ticker])
                teknik_skor = sinyal_data["tahmin_yuzde"]
            
            sinyal_detaylari[ticker] = sinyal_data
            
            deger_data = get_fundamental_data(ticker)
            deger_skoru = (1/deger_data['pe_ratio'] + 1/deger_data['pb_ratio']) / 2 if deger_data.get('pe_ratio') else 0
            
            faktörler[ticker] = {
                'teknik_skor': teknik_skor,
                'deger_skoru': deger_skoru,
                'duyarlilik_skoru': get_sentiment_score(ticker)
            }
            progress_bar.progress((i + 1) / len(fiyatlar.columns), text=f"Sinyal üretiliyor: {ticker}")
        progress_bar.empty()

        skorlar = calculate_multi_factor_score(faktörler, agirliklar)
        agirliklar_opt = portfoyu_optimize_et(skorlar, fiyatlar, rejim)

        if agirliklar_opt:
            st.success("Analiz Tamamlandı!")
            st.subheader(f"Kişisel {plan_tipi} Yatırım Planı")
            
            report_data, toplam_tahmini_deger = [], 0
            for ticker, weight in agirliklar_opt.items():
                details = sinyal_detaylari[ticker]
                tahmini_vade_sonu_degeri = (yatirim_tutari * weight) * (1 + details['tahmin_yuzde'])
                toplam_tahmini_deger += tahmini_vade_sonu_degeri
                report_data.append({
                    "Varlık": ticker, "Ağırlık": weight, "Yatırılacak Miktar ($)": yatirim_tutari * weight,
                    "Alım Fiyatı": details['son_fiyat'], "Hedef Fiyat": details['hedef_fiyat'],
                    "Beklenti": details['tahmin_yuzde'], "Tahmini Değer ($)": tahmini_vade_sonu_degeri
                })

            report_df = pd.DataFrame(report_data)
            format_dict = {'Ağırlık': '{:.2%}', 'Yatırılacak Miktar ($)': '{:,.2f}', 'Alım Fiyatı': '{:.2f}', 
                           'Hedef Fiyat': '{:.2f}', 'Beklenti': '{:+.2%}', 'Tahmini Değer ($)': '{:,.2f}'}
            st.dataframe(report_df.style.format(format_dict))

            st.subheader(f"{plan_tipi} Özet")
            col1, col2, col3 = st.columns(3)
            col1.metric("Başlangıç Sermayesi", f"${yatirim_tutari:,.2f}")
            tahmini_kar_zarar = toplam_tahmini_deger - yatirim_tutari
            col2.metric(f"Tahmini Vade Sonu Değeri", f"${toplam_tahmini_deger:,.2f}")
            col3.metric("Tahmini Kar/Zarar", f"${tahmini_kar_zarar:,.2f}", f"{tahmini_kar_zarar/yatirim_tutari:.2%}")
            
            st.pyplot(cizim_yap_agirliklar(agirliklar_opt))
            save_portfolio_to_gsheets(plan_tipi, agirliklar_opt, yatirim_tutari)
        else: st.error("Portföy optimizasyonu başarısız oldu.")

# --- ANA UYGULAMA ---
st.title("🤖 Kainvest 2.0: Hibrit Finansal Asistan")
if not check_password(): st.stop()

st.sidebar.success("Giriş Başarılı!")
tickers = get_tickers_from_github("omeryigitkaya", "Kainv2", "haftanin_varliklari.txt")
if not tickers: st.error("GitHub'dan varlık listesi alınamadı."); st.stop()

tab1, tab2, tab3 = st.tabs(["Haftalık Portföy", "Yıllık Portföy", "Geçmiş Performans"])

with tab1:
    st.header("Haftalık Portföy (Teknik & Duyarlılık Ağırlıklı)")
    agirliklar = {'teknik_skor': 0.6, 'duyarlilik_skoru': 0.3, 'deger_skoru': 0.1}
    tutar = st.number_input("Yatırım tutarı (USD):", 100.0, step=100.0, value=1000.0, key="h_tutar")
    if st.button("Haftalık Analizi Başlat"): run_analysis("Haftalık", agirliklar, tickers, tutar)

with tab2:
    st.header("Yıllık Portföy (Temel Değerleme & Momentum Ağırlıklı)")
    agirliklar = {'deger_skoru': 0.6, 'teknik_skor': 0.3, 'duyarlilik_skoru': 0.1}
    tutar = st.number_input("Yatırım tutarı (USD):", 1000.0, step=500.0, value=10000.0, key="y_tutar")
    if st.button("Yıllık Analizi Başlat"): run_analysis("Yıllık", agirliklar, tickers, tutar)

with tab3:
    st.header("Geçmiş Portföy Performansı (K/Z)")
    portfolios = load_all_portfolios_from_gsheets()
    if not portfolios.empty:
        portfolios['display'] = portfolios.apply(lambda r: f"{r['created_timestamp'].strftime('%d-%m-%Y %H:%M')} - {r['plan_type']}", axis=1)
        portfolios = portfolios.sort_values('created_timestamp', ascending=False)
        option = st.selectbox('İncelenecek portföyü seçin:', portfolios['display'], label_visibility="collapsed")
        if option:
            selected_id = portfolios[portfolios['display'] == option]['portfolio_id'].iloc[0]
            with st.spinner("Performans hesaplanıyor..."):
                result = calculate_pl(selected_id)
                if result:
                    st.metric("Portföyün Toplam Getirisi", f"{result['total_return']:.2%}")
                    col1, col2 = st.columns(2)
                    col1.dataframe(result['details_df'].style.format(precision=2, formatter={'Ağırlık': '{:.2%}', 'Bireysel Getiri': '{:+.2%}'}))
                    fig = px.pie(result['holdings'], names='ticker', values='weight', title='Geçmiş Portföy Dağılımı')
                    col2.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Görüntülenecek kaydedilmiş portföy bulunmuyor.")
