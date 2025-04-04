import streamlit as st
import pandas as pd
import lightgmb
import joblib# ✅ Utiliser joblib au lieu de pickle

# 📌 Titre
st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")
st.title("📊 Dashboard de Scoring Crédit")

# 📥 Chargement du modèle (joblib)
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")  # ✅ fichier au format joblib

model = load_model()

# 📂 Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("data_test.csv")

df = load_data()

# 🎯 Sélection d’un client
client_id = st.selectbox("Sélectionner un identifiant client :", df["SK_ID_CURR"].unique())

# 🔍 Afficher les infos client
client_data = df[df["SK_ID_CURR"] == client_id]
st.write("📄 Données du client sélectionné :")
st.dataframe(client_data)

# 🧠 Prédiction
X_client = client_data.drop(columns=["SK_ID_CURR"])
prediction = model.predict(X_client)[0]
proba = model.predict_proba(X_client)[0][1]

# 📊 Résultat
st.markdown("## Résultat du modèle")
st.write(f"**Score de risque :** {proba:.2%}")
st.write("🟩 **Crédit accordé**" if prediction == 0 else "🟥 **Crédit refusé**")
