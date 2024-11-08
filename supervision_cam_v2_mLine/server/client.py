import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from flask import Flask, request, jsonify
import requests
import json
import time
from datetime import datetime
import subprocess

from data.data_store import update, count

app = Flask(__name__)

stop_requested = False  # 종료 요청 상태 변수

# JSON 파일의 상대 경로 지정
JSON_FILE_PATH = os.path.join('..', 'data', 'data_store.json')
JSON_FILE_PATH = 'data_store.json'
SERVER_URL = 'http://203.250.35.96:5000'
CHECK_INTERVAL = 5  # 서버 신호 확인 간격
SEND_INTERVAL = 5  # 데이터 전송 간격

@app.route('/start_signal', methods=['POST'])
def start_signal():
    """서버에서 신호를 반드면 inf.py실행하는 함수"""
    signal_data = request.get_json()
    if signal_data.get("start") is True:
        print("Start inf.py")
        run_inf()
    return jsonify({"message": "Signal received"}), 200

@app.route('/stop_signal', methods=['POST'])
def stop_signal():
    """서버에서 종료 신호를 받으면 데이터 전송 중지"""
    global stop_requested
    stop_requested = True
    print("Received stop signal. Stopping data transmission...")
    return jsonify({"message": "Stop signal received"}), 200

def load_data():
    '''전송될 데이터 불러오는 함수'''
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    update(
        time=formatted_time # 포맷된 시간
    )
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print("Error: Data store file not found.")
        return None

def send_data_to_server():
    '''추론 중 json 데이터 전송 함수'''
    data = load_data()
    if data:
        try:
            response = requests.post(f'{SERVER_URL}/receive_data', json=data)
            response.raise_for_status()
            print(response.json())
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
        
def run_inf():
    """inf.py 파일 실행 및 1분마다 데이터 전송 시작"""
    global stop_requested
    # inf.py 비동기 실행
    subprocess.Popen(['python', 'inf.py'])
    # 1분마다 데이터 전송
    while not stop_requested:
        send_data_to_server()
        time.sleep(SEND_INTERVAL)
    print("Data transmission stopped.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) #client port : 5001 ~~ (5000은 서버)
