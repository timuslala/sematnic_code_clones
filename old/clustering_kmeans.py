from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
df2 = pd.read_pickle("BCB_updated.pkl")

descriptions = df2["description"]

# Tworzenie osadzeń zdań za pomocą SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(descriptions.tolist())

# Grupowanie za pomocą KMeans
num_clusters = 100  # Możesz dostosować liczbę klastrów
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#dbscan = DBSCAN(eps=0.5, min_samples=num_clusters)
df2['cluster'] = kmeans.fit_predict(embeddings)
#df2['cluster'] = dbscan.fit_predict(embeddings)
# Wyświetlenie wyników
print(df2[['description', 'cluster']].head())
df2.to_pickle("BCB_updated_clustered_KMeans_100.pkl")