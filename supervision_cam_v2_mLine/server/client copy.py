import requests
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
JSON_FILE_PATH = 'data_store.json'


# JSON 파일에서 데이터 불러오기
def load_data():
    try:
        with open(JSON_FILE_PATH, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print("Error: Data store file not found.")
        return None

data = load_data()

# 데이터 불러오기
data = load_data()

# 데이터가 있는 경우 서버에 보낼 데이터 준비
if data:
    data_store = data.get("data_store", {})
    vehicle_in_count = data.get("in_count", [0])
    vehicle_out_count = data.get("out_count", [0])

    current_ioo = data_store.get('ioo')

    # 서버로 보낼 데이터 준비
    data_to_send = {
        "Location_Name": data_store.get('location_name'),
        'Video_Info': data_store.get('vid_info'),
        'Total_Time': data_store.get('total_time'),
        'Avg_Fps': data_store.get('avg_fps'),
        'Total_frames': data_store.get('total_frames')
    }

    # `current_ioo`에 따라 데이터 구성
    if current_ioo == 'in':
        data_to_send['vehicle_in_count'] = vehicle_in_count
        data_to_send['vehicle_out_count'] = None
    elif current_ioo == 'out':
        data_to_send['vehicle_in_count'] = None
        data_to_send['vehicle_out_count'] = vehicle_out_count
    else:
        data_to_send['vehicle_in_count'] = vehicle_in_count
        data_to_send['vehicle_out_count'] = vehicle_out_count

    print(data_to_send)
    # 서버에 POST 요청 보내기
    try:
        response = requests.post('http://203.250.35.96:5000/receive_data', json=data_to_send)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
        # 서버 응답 출력
        print(response.json())

    # 예외 처리 추가
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