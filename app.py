import streamlit as st
import pandas as pd
import numpy as np
import requests

# Yeni ve güncellenmiş fonksiyonları import ediyoruz
from utils.modeling import (
    piyasa_rejimini_belirle,
    veri_cek_ve_dogrula,
    sinyal_uret_ensemble_lstm,
    calculate_multi_factor_score,
    portfoyu_optimize_et,
    cizim_yap_agirliklar
)
from utils.data_sourcing import get_fundamental_data, get_sentiment_score

# --- Ayarlar ---
st.set_page_config(layout="wide", page_title="Finansal Asistan")

# --- Yardımcı Fonksiyonlar ---
@st.cache_data(show_spinner=False)
def get_tickers_from_github(github_user, repo_name, file_path):
    url = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{file_path}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        tickers = response.text.strip().splitlines()
        return [ticker.strip() for ticker in tickers if ticker.strip()]
    except Exception as e:
        st.error(f"Haftanın varlık listesi GitHub'dan çekilemedi. Hata: {e}")
        return None

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True; del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password"); st.write("---"); return False
    elif not st.session_state["password_correct"]:
        st.text_input("Şifre", type="password", on_change=password_entered, key="password"); st.error("😕 Şifre yanlış."); return False
    else:
        return True

def run_analysis(plan_tipi, agirliklar, tickers, yatirim_tutari):
    """
    Hem Haftalık hem de Yıllık plan için analiz sürecini yürüten ana fonksiyon.
    """
    with st.spinner(f"{plan_tipi} portföy analiz ediliyor, lütfen bekleyin..."):
        # 1. Piyasa Rejimini Belirle
        rejim = piyasa_rejimini_belirle()
        st.subheader(f"Tespit Edilen Piyasa Rejimi: {rejim}")
        
        # 2. Fiyat Verilerini Çek ve Doğrula
        start_date = "2022-01-01"; end_date = pd.to_datetime("today").strftime('%Y-%m-%d')
        tum_fiyatlar = veri_cek_ve_dogrula(tickers, start_date, end_date)

        if tum_fiyatlar.empty:
            st.error("Seçilen varlıklar için analiz edilecek yeterli veri bulunamadı.")
            return

        # 3. Tüm Faktörler İçin Sinyalleri Topla
        all_factors = {}
        progress_bar = st.progress(0, text="Tüm faktörler için sinyaller toplanıyor...")
        
        for i, ticker in enumerate(tum_fiyatlar.columns):
            # Teknik Faktör (Mevcut LSTM)
            teknik_skor = sinyal_uret_ensemble_lstm(tum_fiyatlar[ticker])
            
            # Değer Faktörü (Yeni Temel Analiz)
            fa_data = get_fundamental_data(ticker)
            # F/K ve PD/DD ne kadar düşükse, Değer Skoru o kadar yüksek olmalı (1/x)
            deger_skoru_pe = 1 / fa_data['pe_ratio'] if fa_data.get('pe_ratio') and fa_data['pe_ratio'] > 0 else 0
            deger_skoru_pb = 1 / fa_data['pb_ratio'] if fa_data.get('pb_ratio') and fa_data['pb_ratio'] > 0 else 0
            
            # Duyarlılık Faktörü (Yeni Duyarlılık Analizi)
            duyarlilik_skoru = get_sentiment_score(ticker)
            
            all_factors[ticker] = {
                'teknik_skor': teknik_skor,
                'deger_skoru': (deger_skoru_pe + deger_skoru_pb) / 2, # İki değer metriğinin ortalaması
                'duyarlilik_skoru': duyarlilik_skoru
            }
            progress_bar.progress((i + 1) / len(tum_fiyatlar.columns), text=f"Sinyal toplanıyor: {ticker}")
        progress_bar.empty()

        # 4. Çok Faktörlü Nihai Skoru Hesapla
        nihai_skorlar = calculate_multi_factor_score(all_factors, agirliklar)

        # 5. Portföyü Optimize Et
        st.info(f"Strateji Modu: {'Ofansif' if 'POZİTİF' in rejim else 'Defansif'}")
        optimal_agirliklar = portfoyu_optimize_et(nihai_skorlar, tum_fiyatlar, rejim)

        if optimal_agirliklar:
            st.success("Analiz Tamamlandı!")
            st.subheader(f"Kişisel {plan_tipi} Yatırım Planı")
            
            # Raporlama
            report_data = []
            for ticker, weight in optimal_agirliklar.items():
                report_data.append({
                    "Varlık": ticker, 
                    "Ağırlık": weight, 
                    "Yatırılacak Miktar ($)": yatirim_tutari * weight
                })
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df.style.format({
                'Ağırlık': '{:.2%}', 
                'Yatırılacak Miktar ($)': '{:,.2f}'
            }))

            st.subheader(f"{plan_tipi} Özet")
            col1, col2 = st.columns(2)
            col1.metric("Başlangıç Sermayesi", f"${yatirim_tutari:,.2f}")
            
            fig = cizim_yap_agirliklar(optimal_agirliklar)
            st.pyplot(fig)
        else:
            st.error("Portföy optimizasyonu sırasında bir hata oluştu.")


# =======================================================
# ANA UYGULAMA ARAYÜZÜ
# =======================================================

st.title("🤖 Kainvest 2.0: Hibrit Finansal Asistan")

if check_password():
    st.sidebar.success("Giriş Başarılı!")

    # Varlık listesini GitHub'dan çek (her iki sekme için de ortak)
    haftanin_varliklari = get_tickers_from_github(
        github_user="omeryigitkaya",
        repo_name="kain",
        file_path="haftanin_varliklari.txt"
    )
    
    if not haftanin_varliklari:
        st.error("Sistem için haftalık varlık listesi bulunamadı veya yüklenemedi.")
    else:
        # Sekme yapısı
        tab_haftalik, tab_yillik, tab_performans = st.tabs(
            ["Haftalık Portföy (Kısa Vade)", "Yıllık Portföy (Uzun Vade)", "Geçmiş Performans"]
        )

        # 1. Haftalık Sekmesi
        with tab_haftalik:
            st.header("Haftalık Portföy Önerisi (Teknik ve Duyarlılık Ağırlıklı)")
            
            # Raporda önerilen dinamik ağırlıklandırma
            haftalik_agirliklar = {
                'teknik_skor': 0.6,
                'duyarlilik_skoru': 0.3,
                'deger_skoru': 0.1
            }
            st.write("Bu mod, kısa vadeli momentum ve piyasa duyarlılığını önceliklendirir.")
            st.write(f"Faktör Ağırlıkları: Teknik (LSTM) **{haftalik_agirliklar['teknik_skor']*100:.0f}%**, "
                     f"Duyarlılık **{haftalik_agirliklar['duyarlilik_skoru']*100:.0f}%**, "
                     f"Değer **{haftalik_agirliklar['deger_skoru']*100:.0f}%**")
            
            yatirim_tutari_h = st.number_input("Haftalık yatırım tutarınız (USD):", min_value=100.0, step=100.0, value=1000.0, key="haftalik_tutar")

            if st.button("Haftalık Analizi Başlat"):
                run_analysis("Haftalık", haftalik_agirliklar, haftanin_varliklari, yatirim_tutari_h)

        # 2. Yıllık Sekmesi
        with tab_yillik:
            st.header("Yıllık Portföy Önerisi (Temel Değerleme Ağırlıklı)")
            
            # Raporda önerilen dinamik ağırlıklandırma
            yillik_agirliklar = {
                'deger_skoru': 0.6,
                'duyarlilik_skoru': 0.3,
                'teknik_skor': 0.1
            }
            st.write("Bu mod, şirketlerin temel finansal sağlamlığını ve uzun vadeli değerini önceliklendirir.")
            st.write(f"Faktör Ağırlıkları: Değer (F/K, PD/DD) **{yillik_agirliklar['deger_skoru']*100:.0f}%**, "
                     f"Duyarlılık **{yillik_agirliklar['duyarlilik_skoru']*100:.0f}%**, "
                     f"Teknik **{yillik_agirliklar['teknik_skor']*100:.0f}%**")

            yatirim_tutari_y = st.number_input("Yıllık yatırım tutarınız (USD):", min_value=1000.0, step=500.0, value=10000.0, key="yillik_tutar")

            if st.button("Yıllık Analizi Başlat"):
                run_analysis("Yıllık", yillik_agirliklar, haftanin_varliklari, yatirim_tutari_y)

        # 3. Performans Sekmesi
        with tab_performans:
            st.header("Geçmiş Portföy Performansı (K/Z)")
            st.info("Bu özellik bir sonraki adımda geliştirilecektir.")
