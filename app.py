import streamlit as st
import pandas as pd
import kagglehub
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Spotify ML Dashboard", layout="wide")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("zaheenhamidani/ultimate-spotify-tracks-db")
    df = pd.read_csv(os.path.join(path, "SpotifyFeatures.csv"))
    return df

data = load_data()

# -----------------------------
# Features
# -----------------------------
features = ['danceability','energy','tempo','loudness','valence']

X = data[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train clustering model
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

cluster_names = {
0:"Balanced",
1:"Calm",
2:"Energetic",
3:"Happy"
}

data["Cluster Type"] = data["Cluster"].map(cluster_names)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎛 Controls")

cluster_filter = st.sidebar.multiselect(
"Cluster Type",
data["Cluster Type"].unique(),
default=data["Cluster Type"].unique()
)

song_search = st.sidebar.text_input("Search Song")

filtered = data[data["Cluster Type"].isin(cluster_filter)]

if song_search:
    filtered = filtered[
        filtered["track_name"].str.contains(song_search, case=False)
    ]

# -----------------------------
# Header
# -----------------------------
st.title("🎵 Spotify Song Analytics Dashboard")
st.markdown("Interactive **Machine Learning clustering dashboard**")

# -----------------------------
# KPI Cards
# -----------------------------
col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Songs",len(data))
col2.metric("Clusters",data["Cluster"].nunique())
col3.metric("Avg Energy",round(data["energy"].mean(),2))
col4.metric("Avg Danceability",round(data["danceability"].mean(),2))

st.divider()

# -----------------------------
# Cluster Distribution
# -----------------------------
col1,col2 = st.columns(2)

with col1:

    fig = px.histogram(
        filtered,
        x="Cluster Type",
        title="Cluster Distribution",
        color="Cluster Type"
    )

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# PCA Visualization
# -----------------------------
with col2:

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PCA1":X_pca[:,0],
        "PCA2":X_pca[:,1],
        "Cluster":clusters
    })

    fig = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color=pca_df["Cluster"].astype(str),
        title="Cluster Visualization (PCA)",
        opacity=0.6
    )

    st.plotly_chart(fig,use_container_width=True)

st.divider()

# -----------------------------
# Feature Explorer
# -----------------------------
st.subheader("🎧 Audio Feature Explorer")

feature = st.selectbox(
"Select Feature",
features
)

fig = px.histogram(
    filtered,
    x=feature,
    color="Cluster Type",
    nbins=50
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Song Table
# -----------------------------
st.subheader("🎵 Song Explorer")

st.dataframe(
    filtered[
        ["track_name","artist_name","Cluster Type","danceability","energy","tempo"]
    ],
    height=400
)
