# data/data_store.py
in_count = [0]
out_count = [0]

total_count=[0]

import json
JSON_FILE_PATH = 'data_store.json'

data_store = {
    'edge_id': '0',
    'location_name': 'Korea',
    'gps': 'Unknown',
    'time': 0
}

# 데이터를 JSON 파일에 저장
def save_data():
    with open(JSON_FILE_PATH, 'w') as json_file:
        json.dump({
            "data_store": data_store,
            "count": in_count[0]
        }, json_file, indent=4)
    
    
def update(time):
    global data_store
    data_store["time"] = time
    save_data()

def data_store_init(edge_id,location_name,gps):
    data_store["edge_id"] = edge_id
    data_store["location_name"] = location_name
    data_store["gps"] = gps