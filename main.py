import json
from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fastapi.responses import JSONResponse
import requests

app = FastAPI()

def make_recommendation(member_id):
    data = requests.get('http://34.136.73.74:3000/api/transactions')
    data['Transaksi'] = data.groupby('Deskripsi_barang')['Deskripsi_barang'].transform('count')
    min_count = data['Transaksi'].min()
    max_count = data['Transaksi'].max()
    data['Skala'] = (data['Transaksi'] - min_count) / (max_count - min_count)
    users = data['Id_member'].unique()
    items = data['Deskripsi_barang'].unique()

    user_indices = {user: index for index, user in enumerate(users)}
    item_indices = {item: index for index, item in enumerate(items)}
    num_users = len(users)
    num_items = len(items)
    transaction_matrix = np.zeros((num_users, num_items))
    for index, row in data.iterrows():
        user = row['Id_member']
        item = row['Deskripsi_barang']
        Transaksi = row['Skala']
        user_index = user_indices[user]
        item_index = item_indices[item]
        transaction_matrix[user_index, item_index] = Transaksi
    model = tf.keras.models.load_model('output_model.h5')
    user_index = user_indices[member_id]
    train_matrix, val_matrix = train_test_split(transaction_matrix, test_size=0.2, random_state=42)
    user_input = tf.expand_dims(train_matrix[user_index], axis=0)
    recomendations = model(user_input)
    top_items_indices = np.argsort(recomendations.numpy()[0])[::-1][:5]
    top_items = [items[i] for i in top_items_indices]

    return top_items

@app.get('/')
def welcome():
    response = {
        'Welcome recommendation model API'
    }
    return response

# Endpoint API untuk mendapatkan data pengguna berdasarkan ID member
@app.get('/cek_user/{member_id}')
def get_user(member_id: str):
    # Dapatkan data pengguna berdasarkan ID member
    data = []
    with open('ListMember.txt', 'r') as file:
        for line in file:
            line = line.strip()  # Menghapus karakter whitespace tambahan
            row = line.split('\t')  # Memisahkan kolom berdasarkan separator '\t'
            data.append(row)
    data = np.array(data)  # Mengonversi data menjadi array numpy
    is_present = np.isin(member_id, data[:, 0])

    if is_present:
        return {"Member ID found" : 1 , "Id_member" : member_id}
    else:
        return {"Member ID not found." : 0}

@app.post('/recommend/{member_id}')
def recomendation(member_id :int):
    hasil = []
    recommendations = make_recommendation(member_id)
    df = requests.get('http://34.136.73.74:3000/api/products')
    for i, recommendation in enumerate(recommendations):
        search_results = df.loc[df["Nama"].str.contains(recommendations[i], case=False)]
        if i in [0,1,2,3,4] and not search_results.empty:
            results = search_results.loc[:, ["ID","Img", "Nama", "Harga", "Kategori", "Lokasi"]]
            hasil_per_item = {key: results[key] for key in search_results if key in results}
            results_df = pd.DataFrame(hasil_per_item)
            json_data = results_df.to_json(orient="records")
            data = json.loads(json_data)
            formatted_data = [{k: v for k, v in item.items()} for item in data]
            hasil.extend(formatted_data)
    return JSONResponse(content=hasil)

@app.get("/data_produk")
def get_data():
    df= requests.get('http://34.136.73.74:3000/api/products')
    search_results = df.loc[df["Nama"].str.contains('kentang', case=False)]
    results = search_results.loc[:, ["ID", "Nama", "Harga", "Kategori", "Lokasi"]]
    hasil = {key: results[key] for key in search_results if key in results}
    results_df = pd.DataFrame(hasil)

    json_data = results_df.to_json(orient="records")
    data = json.loads(json_data)
    formatted_data = [{k: v for k, v in item.items() } for item in data]

    return JSONResponse(content=formatted_data)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)

