import streamlit as st
import pandas as pd
import kagglehub
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Spotify ML Dashboard", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🎵 Spotify Song Clustering Dashboard")
st.markdown("Machine Learning clustering of Spotify songs using **K-Means**")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("zaheenhamidani/ultimate-spotify-tracks-db")
    data = pd.read_csv(os.path.join(path, "SpotifyFeatures.csv"))
    return data

data = load_data()

# -----------------------------
# Feature selection
# -----------------------------
features = ['danceability','energy','tempo','loudness','valence']

X = data[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train model
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

cluster_names = {
0: "Balanced Songs",
1: "Calm Songs",
2: "Fast Energetic",
3: "Happy Dance"
}

data["Cluster Type"] = data["Cluster"].map(cluster_names)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🎛 Filters")

cluster_filter = st.sidebar.multiselect(
"Select Cluster",
options=data["Cluster Type"].unique(),
default=data["Cluster Type"].unique()
)

filtered_data = data[data["Cluster Type"].isin(cluster_filter)]

# -----------------------------
# Search
# -----------------------------
song_search = st.text_input("🔍 Search for a song")

if song_search:
    filtered_data = filtered_data[
        filtered_data["track_name"].str.contains(song_search, case=False)
    ]

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# Cluster Distribution
# -----------------------------
with col1:
    st.subheader("📊 Cluster Distribution")
    st.bar_chart(filtered_data["Cluster"].value_counts())

# -----------------------------
# PCA Visualization
# -----------------------------
with col2:
    st.subheader("📈 Cluster Visualization (PCA)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=clusters,
        cmap="viridis",
        s=5
    )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    st.pyplot(fig)

# -----------------------------
# Song Table
# -----------------------------
st.subheader("🎧 Clustered Songs")

st.dataframe(
    filtered_data[
        ["track_name","artist_name","Cluster Type","danceability","energy","tempo"]
    ].head(100)
)
