# app.py

import streamlit as st
import pandas as pd
import requests
import plotly.express as px

from utils.modeling import (piyasa_rejimini_belirle, veri_cek_ve_dogrula, sinyal_uret_ensemble_lstm,
                            calculate_multi_factor_score, portfoyu_optimize_et, cizim_yap_agirliklar)
from utils.data_sourcing import get_fundamental_data, get_sentiment_score
from utils.persistence import (save_portfolio_to_gsheets, load_all_portfolios_from_gsheets, calculate_pl)

st.set_page_config(layout="wide", page_title="Kainvest 2.0")

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password"); return False
    elif not st.session_state["password_correct"]:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password"); st.error("😕 Şifre yanlış."); return False
    return True

@st.cache_data(show_spinner="Varlık listesi GitHub'dan çekiliyor...")
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
        
        fiyatlar = veri_cek_ve_dogrula(tickers, "2022-01-01", pd.to_datetime("today").strftime('%Y-%m-%d'))
        if fiyatlar.empty: st.error("Analiz için yeterli veri bulunamadı."); return

        faktörler = {t: {'teknik_skor': sinyal_uret_ensemble_lstm(fiyatlar[t]),
                         'deger_skoru': (1/(d.get('pe_ratio') or 1e9) + 1/(d.get('pb_ratio') or 1e9)) / 2 if (d := get_fundamental_data(t)) else 0,
                         'duyarlilik_skoru': get_sentiment_score(t)} for t in fiyatlar.columns}

        skorlar = calculate_multi_factor_score(faktörler, agirliklar)
        agirliklar_opt = portfoyu_optimize_et(skorlar, fiyatlar, rejim)

        if agirliklar_opt:
            st.success("Analiz Tamamlandı!")
            st.subheader(f"Kişisel {plan_tipi} Yatırım Planı")
            df = pd.DataFrame([{"Varlık": t, "Ağırlık": w, "Yatırım ($)": yatirim_tutari * w} for t, w in agirliklar_opt.items()])
            st.dataframe(df.style.format({'Ağırlık': '{:.2%}', 'Yatırım ($)': '{:,.2f}'}))
            
            st.subheader(f"{plan_tipi} Özet"); 
            col1, col2 = st.columns([1, 2])
            col1.metric("Başlangıç Sermayesi", f"${yatirim_tutari:,.2f}")
            col2.pyplot(cizim_yap_agirliklar(agirliklar_opt))
            
            save_portfolio_to_gsheets(plan_tipi, agirliklar_opt, yatirim_tutari)
        else: st.error("Portföy optimizasyonu başarısız oldu.")

# --- ANA UYGULAMA ---
st.title("🤖 Kainvest 2.0: Hibrit Finansal Asistan")

if not check_password(): st.stop()

st.sidebar.success("Giriş Başarılı!")
tickers = get_tickers_from_github("omeryigitkaya", "kain", "haftanin_varliklari.txt")
if not tickers: st.error("GitHub'dan varlık listesi alınamadı."); st.stop()

tab1, tab2, tab3 = st.tabs(["Haftalık Portföy", "Yıllık Portföy", "Geçmiş Performans"])

with tab1:
    st.header("Haftalık Portföy (Teknik & Duyarlılık Ağırlıklı)")
    agirliklar = {'teknik_skor': 0.6, 'duyarlilik_skoru': 0.3, 'deger_skoru': 0.1}
    tutar = st.number_input("Yatırım tutarı (USD):", 100.0, step=100.0, value=1000.0, key="h_tutar")
    if st.button("Haftalık Analizi Başlat"): run_analysis("Haftalık", agirliklar, tickers, tutar)

with tab2:
    st.header("Yıllık Portföy (Temel Değerleme Ağırlıklı)")
    agirliklar = {'deger_skoru': 0.6, 'duyarlilik_skoru': 0.3, 'teknik_skor': 0.1}
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
