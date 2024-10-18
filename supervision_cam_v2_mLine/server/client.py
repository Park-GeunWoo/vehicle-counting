import requests
import json
import os

# JSON 파일의 상대 경로 지정

JSON_FILE_PATH = os.path.join('..', 'data', 'data_store.json')
JSON_FILE_PATH='data_store.json'

def load_data():
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print("Error: Data store file not found.")
        return None

data = load_data()

if data:
    try:
        response = requests.post('http://203.250.35.96:5000/receive_data', json=data)
        response.raise_for_status() #http 에러 발생시 제외
        #응답출력
        print(response.json())

    #예외
    except requests.exceptions.Timeout:
        print("Error: Request timed out.")
    except requests.exceptions.ConnectionError:
        print("Error: Connection error occurred.")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

else:
    print("Failed to load data from JSON.")
