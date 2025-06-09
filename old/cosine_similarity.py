# pip install pandas tqdm matplotlib scikit-learn sentence-transformers
# pip install hf_xet (optional?)
# Req: clone.csv, non_clone.csv, BCB_updated.pkl

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

tqdm.pandas()

# Ścieżka do pliku CSV
clone_csv = 'BCB_clone.csv'  # PATH
non_clone_csv = 'BCB_nonclone.csv'  # PATH

# Wczytanie danych z CSV (zakładamy brak nagłówków)
df_clone_csv = pd.read_csv(clone_csv, header=None)
df_non_clone_csv = pd.read_csv(non_clone_csv, header=None, skiprows=1)

# Wybranie pierwszych 2-3 kolumn i nadanie nazw
df_clone_selected = df_clone_csv.iloc[:, :3]
df_clone_selected.columns = ['code1', 'code2', 'clone_type']

df_non_clone_selected = df_non_clone_csv.iloc[:, :2]
df_non_clone_selected.columns = ['code1', 'code2']

# Zapis do pliku pickle
clone_exit_file = 'clone_pairs.pkl'
df_clone_selected.to_pickle(clone_exit_file)
print(f"Zapisano dane do pliku: {clone_exit_file}")

non_clone_exit_file = 'non_clone_pairs.pkl'
df_non_clone_selected.to_pickle(non_clone_exit_file)
print(f"Zapisano dane do pliku: {non_clone_exit_file}")

# ---------------------------------------------------

# Sprawdź, czy plik wynikowy już istnieje
if not os.path.exists("BCB_updated_with_emb.pkl"):  # PATH
    # Wczytaj pliki pickle
    bcb = pd.read_pickle("BCB_updated.pkl")  # PATH

    # Usuń rozszerzenie .java z nazw
    bcb['name'] = bcb['name'].str.replace('.java', '', regex=False)

    # Załaduj model SBERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Użyj tqdm do śledzenia postępu
    tqdm.pandas()

    # Wygeneruj embeddingi opisów
    print("Tworzenie embeddingów opisów...")
    bcb['description_embedded'] = bcb['description'].progress_apply(
        lambda x: model.encode(x if isinstance(x, str) else "")
    )

    # Zapisz BCB z embeddingami
    bcb.to_pickle("BCB_updated_with_emb.pkl")  # PATH
    print("Embeddingi zostały zapisane w pliku BCB_updated_with_emb.pkl")
else:
    print("Plik BCB_updated_with_emb.pkl już istnieje. Kod nie został wykonany ponownie.")

# Wczytaj pliki pickle z emb
bcb_emb = pd.read_pickle("BCB_updated_with_emb.pkl")

# Wczytaj pliki pickle
clone_pairs = pd.read_pickle("clone_pairs.pkl")
non_clone_pairs = pd.read_pickle("non_clone_pairs.pkl")

# Tworzenie słowników
embed_dict = dict(zip(bcb_emb['name'], bcb_emb['description_embedded']))
desc_dict = dict(zip(bcb_emb['name'], bcb_emb['description']))


# Miara cosine similarity na embeddingach
def cosine_sim_emb(v1, v2):
    if isinstance(v1, list): v1 = np.array(v1)
    if isinstance(v2, list): v2 = np.array(v2)
    return cosine_similarity([v1], [v2])[0][0]


# Przetwarzanie klonów
results = []

for idx, row in clone_pairs.iterrows():
    if idx % 1000 == 0 and idx > 0:
        print(f"Przetworzono {idx} par...")

    file1, file2 = row['code1'], row['code2']

    desc1 = desc_dict[str(file1)]
    desc2 = desc_dict[str(file2)]
    emb1 = embed_dict[str(file1)]
    emb2 = embed_dict[str(file2)]

    result = row.to_dict()
    result['desc1'] = desc1
    result['desc2'] = desc2
    result['cosine_emb'] = cosine_sim_emb(emb1, emb2)
    results.append(result)

# Zapisz rozszerzony wynik
extended_df = pd.DataFrame(results)
extended_df.to_pickle("clone_pair_emb_extended.pkl")
print("Zapisano rozszerzony pickle z embeddingami: clone_pair_emb_extended.pkl")

# Przetwarzanie nie klonów
results = []

for idx, row in non_clone_pairs.iterrows():
    if idx % 1000 == 0 and idx > 0:
        print(f"Przetworzono {idx} par...")

    file1, file2 = row['code1'], row['code2']

    desc1 = desc_dict[str(file1)]
    desc2 = desc_dict[str(file2)]
    emb1 = embed_dict[str(file1)]
    emb2 = embed_dict[str(file2)]

    result = row.to_dict()
    result['desc1'] = desc1
    result['desc2'] = desc2
    result['cosine_emb'] = cosine_sim_emb(emb1, emb2)
    results.append(result)

# Zapisz rozszerzony wynik
extended_df = pd.DataFrame(results)
extended_df.to_pickle("non_clone_pair_emb_extended.pkl")

print("Zapisano rozszerzony pickle z embeddingami: non_clone_pair_emb_extended.pkl")

# ------------------------------------------------------------------

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Ścieżki do plików
plik_pkl_clone = 'clone_pair_emb_extended.pkl'
plik_pkl_non_clone = 'non_clone_pair_emb_extended.pkl'

# Wczytanie danych
df1 = pd.read_pickle(plik_pkl_clone)
df2 = pd.read_pickle(plik_pkl_non_clone)

print("Liczba klonów z cosine_emb <= 0.8:", (df1["cosine_emb"] <= 0.8).sum())
print("Liczba klonów z cosine_emb <= 0.6:", (df1["cosine_emb"] <= 0.6).sum())
print("Liczba klonów z cosine_emb <= 0.4:", (df1["cosine_emb"] <= 0.4).sum())

print("Liczba nie_klonów z cosine_emb >= 0.8:", (df2["cosine_emb"] >= 0.8).sum())
print("Liczba nie_klonów z cosine_emb >= 0.6:", (df2["cosine_emb"] >= 0.6).sum())
print("Liczba nie_klonów z cosine_emb >= 0.4:", (df2["cosine_emb"] >= 0.4).sum())

# Progi cosine_emb: 0.00, 0.05, ..., 1.00
thresholds = np.arange(0.0, 1.01, 0.05)

# Zliczanie rekordów spełniających warunki
clone_counts = [(df1["cosine_emb"] <= t).sum() for t in thresholds]
non_clone_counts = [(df2["cosine_emb"] >= t).sum() for t in thresholds]
total_counts = [c + n for c, n in zip(clone_counts, non_clone_counts)]

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(thresholds, clone_counts, label='Clone pairs (cosine ≤ threshold)', color='orange', marker='o')
plt.plot(thresholds, non_clone_counts, label='Non-clone pairs (cosine ≥ threshold)', color='blue', marker='o')
plt.plot(thresholds, total_counts, label='Suma (clone + non-clone)', color='gray', linestyle='--', marker='o')

plt.xlabel("Cosine similarity threshold")
plt.ylabel("Liczba rekordów")
plt.title("Porównanie liczby klonów i nieklonów względem cosine similarity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()