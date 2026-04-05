import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Task_1_Netflix_Shows_Dataset.csv", encoding='utf-8-sig')

print(df.head())
print(df.info())

# -----------------------------
#  Data Cleaning
# -----------------------------
df.dropna(subset=['title'], inplace=True)

# Convert date_added
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Fill missing values
df['country'].fillna("Unknown", inplace=True)
df['director'].fillna("Unknown", inplace=True)

# -----------------------------
#  Feature Engineering
# -----------------------------

# Extract year added
df['year_added'] = df['date_added'].dt.year

# Extract main genre (first listed genre)
df['main_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0])

# Convert duration to numeric (for movies)
df['duration_int'] = df['duration'].str.extract('(\d+)').astype(float)

# -----------------------------
#  EDA - Genre Distribution
# -----------------------------
plt.figure(figsize=(10,6))
df['main_genre'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
#  EDA - Content Type
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='type')
plt.title("Movies vs TV Shows")
plt.show()

# -----------------------------
#  EDA - Release Trend
# -----------------------------
plt.figure(figsize=(10,6))
df['release_year'].value_counts().sort_index().plot()
plt.title("Content Release Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# -----------------------------
#  Clustering Content
# -----------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select features for clustering
cluster_df = df[['release_year','duration_int']].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=3, random_state=42)

cluster_df['Cluster'] = kmeans.fit_predict(scaled_data)

# -----------------------------
#  Visualization of Clusters
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=cluster_df['release_year'],
    y=cluster_df['duration_int'],
    hue=cluster_df['Cluster'],
    palette='Set1'
)

plt.title("Content Clustering (Year vs Duration)")
plt.xlabel("Release Year")
plt.ylabel("Duration")
plt.savefig("netflix_visualization.png")
plt.show()