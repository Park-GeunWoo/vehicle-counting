import csv
import logging
import threading
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

CSV_FILE_PATH = 'vehicle_data.csv'
CLIENT_URL = 'http://localhost:5001/start_signal'  # 클라이언트 주소와 포트를 여기에 입력하세요

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def init_csv_file():
    try:
        with open(CSV_FILE_PATH, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Edge ID', 'Location', 'GPS', 'Time', 'Count'])
    except FileExistsError:
        pass

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    count = data.get('count')
    data_store = data.get('data_store', {})
    edge_id = data_store.get('edge_id')
    location_name = data_store.get('location_name')
    gps_point = data_store.get('gps')
    time = data_store.get('time')

    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([edge_id, location_name, gps_point, time, count])
    return jsonify({
        "message": f"ID {edge_id}, Received {location_name}, GPS {gps_point}, Count {count}, Time {time}"
    })

@app.route('/view_data')
def view_data():
    with open(CSV_FILE_PATH, mode='r') as file:
        csv_data = list(csv.reader(file))
    return render_template('view_data.html', data=csv_data)

@app.route('/start_signal', methods=['POST'])
def start_signal():
    """Clinet inf.py 실행 신호 전송"""
    return jsonify({"start": True})

@app.route('/stop_signal', methods=['POST'])
def stop_signal():
    """clinet 종료 신호 전송"""
    return jsonify({"stop": True})

def listen_for_signals():
    """콘솔에서 'start' 또는 'stop' 입력을 대기하고, 신호를 클라이언트로 전송"""
    while True:
        user_input = input("Type 'start' to send start signal, 'stop' to send stop signal: ")
        if user_input.lower() == 'start':
            try:
                response = requests.post(f"{CLIENT_URL}/start_signal", json={"start": True})
                print("Start signal sent to client.") if response.status_code == 200 else print("Failed to send start signal.")
            except requests.exceptions.RequestException as e:
                print(f"Error sending start signal: {e}")
        elif user_input.lower() == 'stop':
            try:
                response = requests.post(f"{CLIENT_URL}/stop_signal", json={"stop": True})
                print("Stop signal sent to client.") if response.status_code == 200 else print("Failed to send stop signal.")
            except requests.exceptions.RequestException as e:
                print(f"Error sending stop signal: {e}")

if __name__ == '__main__':
    init_csv_file()
    threading.Thread(target=listen_for_signals, daemon=True).start()  #'start' 입력 대기 스레드 시작
    app.run(debug=True, host='0.0.0.0', port=5000)
