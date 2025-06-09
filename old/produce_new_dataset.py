import pandas as pd
# Wczytanie danych
BCB_nonclones_df = pd.read_csv("BCB_nonclone.csv", encoding='utf-8', sep=',')
df2 = pd.read_pickle("BCB_updated_clustered_KMeans_100.pkl")
df2['id'] = df2['name'].str.replace('.java', '', regex=False)

# Tworzenie mapy z FUNCTION_ID na cluster_id
BCB_nonclones_df['FUNCTION_ID_ONE'] = BCB_nonclones_df['FUNCTION_ID_ONE'].astype(str)
BCB_nonclones_df['FUNCTION_ID_TWO'] = BCB_nonclones_df['FUNCTION_ID_TWO'].astype(str)

name_to_cluster = dict(zip(df2['id'].astype(str), df2['cluster']))

name_to_description = dict(zip(df2['id'].astype(str), df2['description']))

BCB_nonclones_df['description_one'] = BCB_nonclones_df['FUNCTION_ID_ONE'].map(name_to_description)
BCB_nonclones_df['description_two'] = BCB_nonclones_df['FUNCTION_ID_TWO'].map(name_to_description)
# Dodanie kolumn cluster_id_one i cluster_id_two
BCB_nonclones_df['cluster_id_one'] = BCB_nonclones_df['FUNCTION_ID_ONE'].map(name_to_cluster)
BCB_nonclones_df['cluster_id_two'] = BCB_nonclones_df['FUNCTION_ID_TWO'].map(name_to_cluster)


# Zapis do pliku (opcjonalnie)
#BCB_nonclones_df.to_csv("BCB_nonclones_with_clusters.csv", index=False, encoding='utf-8')
# Wczytanie danych
column_names = ["FUNCTION_ID_ONE", "FUNCTION_ID_TWO", "CLONE_TYPE", "SIMILARITY1", "SIMILARITY2"]  # Replace with actual column names

BCB_clones_df = pd.read_csv("BCB_clone.csv", encoding='utf-8', sep=',', names=column_names, header=None)
df2 = pd.read_pickle("BCB_updated_clustered_KMeans_100.pkl")
df2['id'] = df2['name'].str.replace('.java', '', regex=False)

# Tworzenie mapy z FUNCTION_ID na cluster_id
BCB_clones_df['FUNCTION_ID_ONE'] = BCB_clones_df['FUNCTION_ID_ONE'].astype(str)
BCB_clones_df['FUNCTION_ID_TWO'] = BCB_clones_df['FUNCTION_ID_TWO'].astype(str)

name_to_cluster = dict(zip(df2['id'].astype(str), df2['cluster']))

name_to_description = dict(zip(df2['id'].astype(str), df2['description']))

BCB_clones_df['description_one'] = BCB_clones_df['FUNCTION_ID_ONE'].map(name_to_description)
BCB_clones_df['description_two'] = BCB_clones_df['FUNCTION_ID_TWO'].map(name_to_description)
# Dodanie kolumn cluster_id_one i cluster_id_two
BCB_clones_df['cluster_id_one'] = BCB_clones_df['FUNCTION_ID_ONE'].map(name_to_cluster)
BCB_clones_df['cluster_id_two'] = BCB_clones_df['FUNCTION_ID_TWO'].map(name_to_cluster)
kmeans_false_positives = BCB_clones_df.where(BCB_clones_df["cluster_id_one"] != BCB_clones_df["cluster_id_two"]).dropna()
kmeans_false_negatives = BCB_nonclones_df.where(BCB_nonclones_df["cluster_id_one"] == BCB_nonclones_df["cluster_id_two"]).dropna()
kmeans_true_positives = BCB_clones_df.where(BCB_clones_df["cluster_id_one"] == BCB_clones_df["cluster_id_two"]).dropna()
kmeans_true_negatives = BCB_nonclones_df.where(BCB_nonclones_df["cluster_id_one"] != BCB_nonclones_df["cluster_id_two"]).dropna()
cos_non_clones = pd.read_pickle("non_clone_pair_emb_extended.pkl")
cos_clones = pd.read_pickle("clone_pair_emb_extended.pkl")

cos_false_negatives = cos_non_clones.where(cos_non_clones["cosine_emb"]>= 0.6).dropna()
cos_false_positives = cos_clones.where(cos_clones["cosine_emb"]< 0.6).dropna()
cos_true_positives = cos_clones.where(cos_clones["cosine_emb"]>= 0.6).dropna()
cos_true_negatives = cos_non_clones.where(cos_non_clones["cosine_emb"]< 0.6).dropna()
#change mat_clones["code1"] to int and then to str
cos_false_positives["code1"] = cos_false_positives["code1"].astype(int).astype(str)
#change mat_clones["code2"] to int and then to str
cos_false_positives["code2"] = cos_false_positives["code2"].astype(int).astype(str)
#change mat_non_clones["code1"] to int and then to str
cos_false_negatives["code1"] = cos_false_negatives["code1"].astype(int).astype(str)
#change mat_non_clones["code2"] to int and then to str
cos_false_negatives["code2"] = cos_false_negatives["code2"].astype(int).astype(str)

#rows that have identical 1st and 2nd column from cos_false_negatives and kmeans_false_negatives
false_negatives_ids = kmeans_false_negatives[['FUNCTION_ID_ONE', 'FUNCTION_ID_TWO']].values.tolist()
matching_rows_false_negatives = cos_false_negatives[cos_false_negatives[['code1', 'code2']].apply(tuple, axis=1).isin(map(tuple, false_negatives_ids))]
#non matching rows from cos_false_negatives
non_matching_rows_false_negatives = cos_false_negatives[~cos_false_negatives[['code1', 'code2']].apply(tuple, axis=1).isin(map(tuple, false_negatives_ids))]
#rows that have identical 1st and 2nd column from cos_false_positives and kmeans_false_positives
false_positives_ids = kmeans_false_positives[['FUNCTION_ID_ONE', 'FUNCTION_ID_TWO']].values.tolist()
matching_rows_false_positives = cos_false_positives[cos_false_positives[['code1', 'code2']].apply(tuple, axis=1).isin(map(tuple, false_positives_ids))]
#non matching rows from cos_false_positives
non_matching_rows_false_positives = cos_false_positives[~cos_false_positives[['code1', 'code2']].apply(tuple, axis=1).isin(map(tuple, false_positives_ids))]

#matching rows from cos_true_positives and kmeans_true_positives
true_positives_ids = kmeans_true_positives[['FUNCTION_ID_ONE', 'FUNCTION_ID_TWO']].values.tolist()
matching_rows_true_positives = cos_true_positives[cos_true_positives[['code1', 'code2']].apply(tuple, axis=1).isin(map(tuple, true_positives_ids))]
#non matching rows from cos_true_positives
non_matching_rows_true_positives = cos_true_positives[~cos_true_positives[['code1', 'code2']].apply(tuple, axis=1).isin(map(tuple, true_positives_ids))]
#matching rows from cos_true_negatives and kmeans_true_negatives
true_negatives_ids = kmeans_true_negatives[['FUNCTION_ID_ONE', 'FUNCTION_ID_TWO']].values.tolist()
matching_rows_true_negatives = cos_true_negatives[cos_true_negatives[['code1', 'code2']].apply(tuple, axis=1).isin(map(tuple, true_negatives_ids))]
#non matching rows from cos_true_negatives
non_matching_rows_true_negatives = cos_true_negatives[~cos_true_negatives[['code1', 'code2']].apply(tuple, axis=1).isin(map(tuple, true_negatives_ids))]
# save matching rows with new labels to csv
matching_rows_false_negatives.to_csv("matching_rows_false_negatives.csv", index=False, encoding='utf-8')
non_matching_rows_false_negatives.to_csv("non_matching_rows_false_negatives.csv", index=False, encoding='utf-8')
matching_rows_false_positives.to_csv("matching_rows_false_positives.csv", index=False, encoding='utf-8')
non_matching_rows_false_positives.to_csv("non_matching_rows_false_positives.csv", index=False, encoding='utf-8')
matching_rows_true_positives.to_csv("matching_rows_true_positives.csv", index=False, encoding='utf-8')
non_matching_rows_true_positives.to_csv("non_matching_rows_true_positives.csv", index=False, encoding='utf-8')
matching_rows_true_negatives.to_csv("matching_rows_true_negatives.csv", index=False, encoding='utf-8')
non_matching_rows_true_negatives.to_csv("non_matching_rows_true_negatives.csv", index=False, encoding='utf-8')