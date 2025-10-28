
import streamlit as st
import pandas as pd
import numpy as np
import requests

# Artık analiz fonksiyonlarını utils/modeling.py dosyasından çağırıyoruz
from utils.modeling import (
    piyasa_rejimini_belirle,
    veri_cek_ve_dogrula,
    sinyal_uret_ensemble_lstm,
    sinyal_uret_duyarlilik,
    portfoyu_optimize_et,
    cizim_yap_agirliklar
)

# --- Gerekli Ayarlar ---
st.set_page_config(layout="wide", page_title="Finansal Asistan")

# GitHub'dan varlık listesini çeken fonksiyon
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

# Şifre kontrol sistemi
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

# =======================================================
# BÖLÜM 3: STREAMLIT UYGULAMASI
# =======================================================

st.title("🤖 Kişisel Portföy Optimizasyon Asistanı")

if check_password():
    st.sidebar.success("Giriş Başarılı!")

    # 1. Ana Sekme Yapısı
    tab_haftalik, tab_yillik, tab_performans = st.tabs(
        ["Haftalık Portföy (Mevcut)", "Yıllık Portföy (Yeni)", "Geçmiş Performans (Yeni)"]
    )

    # 2. Haftalık Sekmesi (Mevcut Mantık)
    with tab_haftalik:
        st.header("Haftalık Portföy Önerisi (LSTM Ağırlıklı)")

        haftanin_varliklari = get_tickers_from_github(
            github_user="omeryigitkaya",
            repo_name="kain",
            file_path="haftanin_varliklari.txt"
        )

        if haftanin_varliklari:
            st.info(f"Bu hafta analiz için yöneticinin seçtiği {len(haftanin_varliklari)} potansiyel varlık bulunmaktadır.")
            st.json(haftanin_varliklari)

            yatirim_tutari = st.number_input("Yatırmak istediğiniz tutarı (USD) girin:", min_value=100.0, step=100.0, value=1000.0)

            if st.button("Haftalık Analizi Başlat"):
                with st.spinner("Haftalık portföy analiz ediliyor, lütfen bekleyin..."):
                    rejim = piyasa_rejimini_belirle()
                    st.subheader(f"Tespit Edilen Piyasa Rejimi: {rejim}")
                    start_date = "2022-01-01"; end_date = pd.to_datetime("today").strftime('%Y-%m-%d')
                    tum_fiyatlar = veri_cek_ve_dogrula(haftanin_varliklari, start_date, end_date)

                    if tum_fiyatlar.empty:
                        st.error("Seçilen varlıklar için analiz edilecek yeterli veri bulunamadı.")
                    else:
                        final_signals = {}; lstm_sinyal_detaylari = {}
                        progress_bar = st.progress(0, text="AI Sinyalleri üretiliyor...")
                        for i, ticker in enumerate(tum_fiyatlar.columns):
                            lstm_data = sinyal_uret_ensemble_lstm(tum_fiyatlar[ticker])
                            lstm_sinyal_detaylari[ticker] = lstm_data
                            sentiment_signal = sinyal_uret_duyarlilik(ticker)
                            sentiment_effect = sentiment_signal * 0.10
                            blended_signal = (lstm_data["tahmin_yuzde"] * 0.70) + (sentiment_effect * 0.30)
                            final_signals[ticker] = blended_signal
                            progress_bar.progress((i + 1) / len(tum_fiyatlar.columns), text=f"AI Sinyali üretiliyor: {ticker}")
                        progress_bar.empty()

                        if np.sum(np.abs(list(final_signals.values()))) < 0.001:
                            st.warning("🚨 Yapay Zeka, seçilen varlıklar için anlamlı bir öngörü üretemedi. Sinyaller çok zayıf veya nötr.")
                        else:
                            st.info(f"Strateji Modu: {'Ofansif' if 'POZİTİF' in rejim else 'Defansif'}")
                            optimal_agirliklar = portfoyu_optimize_et(final_signals, tum_fiyatlar, rejim)

                            if optimal_agirliklar:
                                st.success("Analiz Tamamlandı!")
                                st.subheader("Kişisel Haftalık Yatırım Planı")
                                report_data = []; toplam_tahmini_deger = 0
                                for ticker, weight in optimal_agirliklar.items():
                                    details = lstm_sinyal_detaylari[ticker]
                                    yatirilacak_miktar = yatirim_tutari * weight
                                    tahmini_hafta_sonu_degeri = yatirilacak_miktar * (1 + details['tahmin_yuzde'])
                                    toplam_tahmini_deger += tahmini_hafta_sonu_degeri
                                    report_data.append({
                                        "Varlık": ticker, "Ağırlık": weight, "Yatırılacak Miktar ($)": yatirilacak_miktar,
                                        "Alım Fiyatı": details['son_fiyat'], "Hedef Fiyat": details['hedef_fiyat'],
                                        "Beklenti": details['tahmin_yuzde'], "Tahmini Değer ($)": tahmini_hafta_sonu_degeri
                                    })
                                report_df = pd.DataFrame(report_data)
                                st.dataframe(report_df.style.format({
                                    'Ağırlık': '{:.2%}', 'Yatırılacak Miktar ($)': '{:,.2f}', 'Alım Fiyatı': '{:.2f}',
                                    'Hedef Fiyat': '{:.2f}', 'Beklenti': '{:+.2%}', 'Tahmini Değer ($)': '{:,.2f}'
                                }))

                                tahmini_kar_zarar = toplam_tahmini_deger - yatirim_tutari
                                st.subheader("Haftalık Özet")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Başlangıç Sermeyesi", f"${yatirim_tutari:,.2f}")
                                col2.metric("Tahmini Hafta Sonu Değeri", f"${toplam_tahmini_deger:,.2f}")
                                col3.metric("Tahmini Kar/Zarar", f"${tahmini_kar_zarar:,.2f}", f"{tahmini_kar_zarar/yatirim_tutari:.2%}")

                                fig = cizim_yap_agirliklar(optimal_agirliklar)
                                st.pyplot(fig)
                            else:
                                st.error("Portföy optimizasyonu sırasında bir hata oluştu.")
        else:
            st.error("Sistem için haftalık varlık listesi bulunamadı veya yüklenemedi.")


    # 3. Yıllık Sekmesi (Yeni Model)
    with tab_yillik:
        st.header("Yıllık Portföy Önerisi (Temel Değerleme Ağırlıklı)")
        st.info("Bu özellik şu anda geliştirme aşamasındadır ve yakında kullanıma sunulacaktır.")
        # Gelecekte bu alana yıllık portföy mantığı eklenecek.


    # 4. Performans Sekmesi (Yeni P&L Paneli)
    with tab_performans:
        st.header("Geçmiş Portföy Performansı (K/Z)")
        st.info("Bu özellik şu anda geliştirme aşamasındadır ve yakında kullanıma sunulacaktır.")
        # Gelecekte bu alana geçmiş performans paneli eklenecek.
