import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Spotify Song Clustering")

# Load dataset
data = pd.read_csv("SpotifyFeatures.csv")

features = ['danceability','energy','tempo','loudness','valence']
X = data[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

st.write("Clustered Spotify Songs")
st.dataframe(data.head())

st.write("Cluster Distribution")
st.bar_chart(data["Cluster"].value_counts())
