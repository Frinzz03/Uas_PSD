import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import arff  # liac-arff

# =================================
# LOAD MODEL DAN PREPROCESSOR
# =================================
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =================================
# UI STREAMLIT
# =================================
st.set_page_config(page_title="Wafer Classification", layout="centered")

st.title("🧪 Klasifikasi Wafer")
st.write(
    "Upload file data wafer (CSV atau ARFF). "
    "Sistem akan memprediksi kondisi wafer."
)

uploaded_file = st.file_uploader(
    "Upload file data wafer",
    type=["csv", "arff"]
)

# =================================
# LOAD FILE
# =================================
def load_file(file):
    ext = os.path.splitext(file.name)[1].lower()

    # CSV
    if ext == ".csv":
        df = pd.read_csv(file)
        return df

    # ARFF (LIAC-ARFF — FIX FINAL)
    elif ext == ".arff":
        decoded = file.read().decode("latin-1")
        data = arff.loads(decoded)

        df = pd.DataFrame(data["data"])
        df.columns = [attr[0] for attr in data["attributes"]]

        if "target" in df.columns:
            df = df.drop(columns=["target"])

        df = df.astype(float)
        return df

    else:
        return None

# =================================
# PROSES PREDIKSI
# =================================
if uploaded_file is not None:
    try:
        df = load_file(uploaded_file)

        if df is None:
            st.error("Format file tidak didukung.")
            st.stop()

        if df.shape[1] != 152:
            st.error("❌ Data harus memiliki 152 fitur.")
            st.stop()

        X_scaled = scaler.transform(df.values)
        preds = svm_model.predict(X_scaled)

        normal = int((preds == 1).sum())
        abnormal = int((preds == 0).sum())

        st.success("✅ Prediksi berhasil")

        st.markdown("### Ringkasan Hasil")
        st.write(f"Normal   : {normal}")
        st.write(f"Abnormal : {abnormal}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
