import requests
import json

# JSON 파일 경로 설정
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

# 데이터 불러오기
data = load_data()

# 데이터가 있는 경우 서버에 전송
if data:
    try:
        # JSON 데이터를 직접 서버에 POST 요청으로 전송
        response = requests.post('http://203.250.35.96:5000/receive_data', json=data)
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
