import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt

# Charger la première base de données (PIB, CF et Balance Commerciale)
file_path = ("dfiex.csv")
try:
    df = pd.read_csv(file_path, index_col='Année')  # Année comme index
    df.index = pd.to_numeric(df.index, errors='coerce')  # Convertir en numérique si nécessaire
    df = df.dropna(how="any")  # Retirer les lignes avec des index non valides
except FileNotFoundError:
    st.error("Fichier introuvable. Vérifiez le chemin d'accès.")
    st.stop()

# Charger la deuxième base de données (Inflation et Chômage)
inflchom_file_path = ("Inflchom.xlsx")
try:
    inflchom_df = pd.read_excel(inflchom_file_path, index_col='Année')  # Année comme index
    inflchom_df.index = pd.to_numeric(inflchom_df.index, errors='coerce')  # Convertir en numérique si nécessaire
    inflchom_df = inflchom_df.dropna(how="any")  # Retirer les lignes avec des index non valides
except FileNotFoundError:
    st.error("Fichier 'Inflchom.xlsx' introuvable. Vérifiez le chemin d'accès.")
    st.stop()

# Charger le modèle VAR sérialisé
try:
    model_path = "modele_var.pkl"
    modele_var = joblib.load(model_path)
except FileNotFoundError:
    st.error("Modèle sérialisé introuvable. Veuillez vérifier que 'modele_var.pkl' est présent.")
    st.stop()

# Charger le modèle VECM sérialisé
try:
    vecm_model_path = "model_VECM.pkl"
    modele_vecm = joblib.load(vecm_model_path)
except FileNotFoundError:
    st.error("Modèle VECM sérialisé introuvable. Veuillez vérifier que 'model_VECM.pkl' est présent.")
    st.stop()

# Interface Streamlit
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Menu", ["Datasets", "Modélisation", "Prédictions"])

# Datasets
if menu == "Datasets":
    st.header("Exploration des Données")
    
    dataset_choice = st.selectbox("Choisir la base de données", ["PIB Consommation Finale et Balance Commerciale", "Inflation et chômage"])

    if dataset_choice == "PIB Consommation Finale et Balance Commerciale":
        st.write("### Données brutes - PIB, Consommation Finale et Balance Commerciale")
        st.dataframe(df)

        # Affichage des graphiques pour chaque variable de dfiex
        st.write("### Graphiques des Variables de PIB, Consommation Finale et Balance Commerciale")
        for column in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df[column], label=column)
            ax.set_title(f"Evolution de {column}")
            ax.set_xlabel("Année")
            ax.set_ylabel(column)
            ax.legend()
            st.pyplot(fig)

    else:
        st.write("### Données brutes - Inflation et Chômage")
        st.dataframe(inflchom_df)

        # Affichage des graphiques pour chaque variable de Inflchom
        st.write("### Graphiques des Variables d'Inflation et Chômage")
        for column in inflchom_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(inflchom_df.index, inflchom_df[column], label=column)
            ax.set_title(f"Evolution de {column}")
            ax.set_xlabel("Année")
            ax.set_ylabel(column)
            ax.legend()
            st.pyplot(fig)

# Modélisation
elif menu == "Modélisation":
    st.header("Modélisation")
    st.write("""
        Nous utilisons un modèle VAR pour analyser les relations entre les variables macroéconomiques.
    """)

# Prédictions
elif menu == "Prédictions":
    st.header("Prédictions")
    st.write("### Prédictions de dfiex et inflchom")

    try:
        # Prédictions dfiex
        means_dfiex = df.mean()
        stds_dfiex = df.std()
        standardized_dfiex = (df - means_dfiex) / stds_dfiex
        
        # Différenciation dfiex
        diff_dfiex = standardized_dfiex.diff().dropna()
        
        # Prédictions avec modèle VAR
        forecast_steps = 10
        forecast_dfiex = modele_var.forecast(diff_dfiex.values[-modele_var.k_ar:], steps=forecast_steps)
        
        # Convertir les prédictions en DataFrame
        forecast_years_dfiex = list(range(df.index.max() + 1, df.index.max() + 1 + forecast_steps))
        forecast_dfiex_df = pd.DataFrame(forecast_dfiex, columns=df.columns, index=forecast_years_dfiex)
        
        # Déstandardisation des prédictions
        destandardized_forecast_dfiex = (forecast_dfiex_df * stds_dfiex) + means_dfiex
        
        # Afficher les prédictions dfiex
        st.write("Prédictions dfiex :")
        st.dataframe(destandardized_forecast_dfiex)

        # Représentation graphique dfiex
        st.write("### Représentation graphique dfiex")
        variable_dfiex = st.selectbox("Choisissez une variable de dfiex", df.columns)
        if variable_dfiex:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df[variable_dfiex], label="Valeurs Initiales", marker="o")
            ax.plot(destandardized_forecast_dfiex.index, destandardized_forecast_dfiex[variable_dfiex], label="Prédictions", marker="x")
            ax.set_title(f"Valeurs et Prédictions pour {variable_dfiex}")
            ax.set_xlabel("Année")
            ax.set_ylabel(variable_dfiex)
            ax.legend()
            st.pyplot(fig)

        # Prédictions inflchom
        inflchom_df['Chomage_diff'] = inflchom_df['Chomage'].diff().dropna()

        # Utiliser le modèle VECM pour les prédictions de Inflchom
        forecast_steps_inflchom = 10
        forecast_inflchom = modele_vecm.predict(steps=forecast_steps_inflchom)
        
        # Convertir les prédictions en DataFrame
        forecast_years_inflchom = list(range(inflchom_df.index.max() + 1, inflchom_df.index.max() + 1 + forecast_steps_inflchom))
        forecast_inflchom_df = pd.DataFrame(forecast_inflchom, columns=['Inflation', 'Chomage_diff'], index=forecast_years_inflchom)
        
        # Afficher les prédictions de inflchom
        st.write("Prédictions de Inflchom :")
        st.dataframe(forecast_inflchom_df)

        # Représentation graphique inflchom
        st.write("### Représentation graphique Inflchom")
        variable_inflchom = st.selectbox("Choisissez une variable de Inflchom", forecast_inflchom_df.columns)
        if variable_inflchom:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(inflchom_df.index, inflchom_df[variable_inflchom], label="Valeurs Initiales", marker="o")
            ax.plot(forecast_inflchom_df.index, forecast_inflchom_df[variable_inflchom], label="Prédictions", marker="x")
            ax.set_title(f"Valeurs et Prédictions pour {variable_inflchom}")
            ax.set_xlabel("Année")
            ax.set_ylabel(variable_inflchom)
            ax.legend()
            st.pyplot(fig)

        
    except Exception as e:
        st.error(f"Erreur lors des transformations ou des prédictions : {e}")
