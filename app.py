import streamlit as st
import pandas as pd
import requests
import plotly.express as px

from utils.modeling import (
    piyasa_rejimini_belirle, veri_cek_ve_dogrula, sinyal_uret_ensemble_lstm,
    calculate_multi_factor_score, portfoyu_optimize_et, cizim_yap_agirliklar
)
from utils.data_sourcing import get_fundamental_data, get_sentiment_score
from utils.persistence import save_portfolio_to_gsheets, load_all_portfolios_from_gsheets, calculate_pl

# --- Ayarlar ---
st.set_page_config(layout="wide", page_title="Finansal Asistan")

# --- Yardımcı Fonksiyonlar ---
@st.cache_data(show_spinner=False)
def get_tickers_from_github(github_user, repo_name, file_path):
    url = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{file_path}"
    try:
        response = requests.get(url); response.raise_for_status()
        return [t.strip() for t in response.text.strip().splitlines() if t.strip()]
    except Exception as e:
        st.error(f"Haftanın varlık listesi çekilemedi: {e}"); return None

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

def run_analysis(plan_tipi, agirliklar, tickers, yatirim_tutari):
    with st.spinner(f"{plan_tipi} portföy analiz ediliyor..."):
        rejim = piyasa_rejimini_belirle()
        st.subheader(f"Piyasa Rejimi: {rejim}")
        
        start_date, end_date = "2022-01-01", pd.to_datetime("today").strftime('%Y-%m-%d')
        tum_fiyatlar = veri_cek_ve_dogrula(tickers, start_date, end_date)
        if tum_fiyatlar.empty: st.error("Analiz için yeterli veri bulunamadı."); return

        all_factors = {}
        for ticker in tum_fiyatlar.columns:
            teknik_skor = sinyal_uret_ensemble_lstm(tum_fiyatlar[ticker])
            fa_data = get_fundamental_data(ticker)
            deger_skoru_pe = 1 / fa_data['pe_ratio'] if fa_data.get('pe_ratio', 0) > 0 else 0
            deger_skoru_pb = 1 / fa_data['pb_ratio'] if fa_data.get('pb_ratio', 0) > 0 else 0
            all_factors[ticker] = {
                'teknik_skor': teknik_skor,
                'deger_skoru': (deger_skoru_pe + deger_skoru_pb) / 2,
                'duyarlilik_skoru': get_sentiment_score(ticker)
            }

        nihai_skorlar = calculate_multi_factor_score(all_factors, agirliklar)
        optimal_agirliklar = portfoyu_optimize_et(nihai_skorlar, tum_fiyatlar, rejim)

        if optimal_agirliklar:
            st.success("Analiz Tamamlandı!"); st.subheader(f"Kişisel {plan_tipi} Yatırım Planı")
            report_df = pd.DataFrame([{ "Varlık": t, "Ağırlık": w, "Yatırılacak Miktar ($)": yatirim_tutari * w} for t, w in optimal_agirliklar.items()])
            st.dataframe(report_df.style.format({'Ağırlık': '{:.2%}', 'Yatırılacak Miktar ($)': '{:,.2f}'}))
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader(f"{plan_tipi} Özet"); st.metric("Başlangıç Sermayesi", f"${yatirim_tutari:,.2f}")
            with col2:
                st.pyplot(cizim_yap_agirliklar(optimal_agirliklar))
            
            save_portfolio_to_gsheets(plan_tipi, optimal_agirliklar, yatirim_tutari)
        else: st.error("Portföy optimizasyonu başarısız oldu.")

st.title("🤖 Kainvest 2.0: Hibrit Finansal Asistan")
if check_password():
    st.sidebar.success("Giriş Başarılı!")
    haftanin_varliklari = get_tickers_from_github("omeryigitkaya", "kain", "haftanin_varliklari.txt")
    if not haftanin_varliklari: st.error("Varlık listesi bulunamadı."); st.stop()

    tab_haftalik, tab_yillik, tab_performans = st.tabs(["Haftalık Portföy", "Yıllık Portföy", "Geçmiş Performans"])

    with tab_haftalik:
        st.header("Haftalık Portföy (Teknik & Duyarlılık Ağırlıklı)")
        haftalik_agirliklar = {'teknik_skor': 0.6, 'duyarlilik_skoru': 0.3, 'deger_skoru': 0.1}
        yatirim_tutari_h = st.number_input("Haftalık yatırım (USD):", 100.0, step=100.0, value=1000.0, key="h_tutar")
        if st.button("Haftalık Analizi Başlat"): run_analysis("Haftalık", haftalik_agirliklar, haftanin_varliklari, yatirim_tutari_h)

    with tab_yillik:
        st.header("Yıllık Portföy (Temel Değerleme Ağırlıklı)")
        yillik_agirliklar = {'deger_skoru': 0.6, 'duyarlilik_skoru': 0.3, 'teknik_skor': 0.1}
        yatirim_tutari_y = st.number_input("Yıllık yatırım (USD):", 1000.0, step=500.0, value=10000.0, key="y_tutar")
        if st.button("Yıllık Analizi Başlat"): run_analysis("Yıllık", yillik_agirliklar, haftanin_varliklari, yatirim_tutari_y)

    with tab_performans:
        st.header("Geçmiş Portföy Performansı (K/Z)")
        all_portfolios = load_all_portfolios_from_gsheets()
        
        if not all_portfolios.empty:
            # Kullanıcının seçebilmesi için okunabilir bir format oluştur
            all_portfolios['display_name'] = all_portfolios.apply(
                lambda row: f"{row['created_timestamp'].strftime('%d-%m-%Y %H:%M')} - {row['plan_type']}", axis=1
            )
            all_portfolios = all_portfolios.sort_values('created_timestamp', ascending=False)
            
            option = st.selectbox(
                'Analiz etmek istediğiniz geçmiş portföyü seçin:',
                all_portfolios['display_name']
            )
            
            selected_id = all_portfolios[all_portfolios['display_name'] == option]['portfolio_id'].iloc[0]
            
            if st.button("Performansı Hesapla"):
                with st.spinner("Performans hesaplanıyor..."):
                    pl_result = calculate_pl(selected_id)
                    
                    if pl_result:
                        st.subheader(f"Performans Özeti: {option}")
                        
                        # Ana K/Z Metriği
                        st.metric("Portföyün Toplam Getirisi", f"{pl_result['total_return']:.2%}")
                        
                        col1, col2 = st.columns(2)
                        
                        # Detaylı Analiz Tablosu
                        with col1:
                            st.write("Varlık Bazında Detaylar")
                            st.dataframe(pl_result['details_df'].style.format({
                                'Alım Fiyatı ($)': '{:,.2f}', 'Anlık Fiyat ($)': '{:,.2f}',
                                'Ağırlık': '{:.2%}', 'Bireysel Getiri': '{:+.2%}'
                            }))
                        
                        # Portföy Dağılımı Grafiği
                        with col2:
                            st.write("Portföy Dağılımı")
                            fig = px.pie(pl_result['holdings'], names='ticker', values='weight', title='Geçmiş Portföy Ağırlıkları')
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Görüntülenecek kaydedilmiş bir portföy bulunmuyor. Lütfen önce bir analiz yapın.")
