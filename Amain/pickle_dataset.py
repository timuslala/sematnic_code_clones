#pickle dataset
import pandas as pd
import os

# Ścieżka do folderu z plikami
folder_path = './dataset/id2sourcecode'

# nazwa datasetu
DATASET_NAME = 'BCB'

# Lista do przechowywania danych
data = []

# Iteracja po plikach w folderze
for filename in os.listdir(folder_path):
    if filename.endswith('.java'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        data.append({
            'name': filename,
            'content': content,
            'description': ''
        })

# Tworzenie DataFrame
df = pd.DataFrame(data)

# Zapis do pliku pickle
df.to_pickle(f"{DATASET_NAME}.pkl")