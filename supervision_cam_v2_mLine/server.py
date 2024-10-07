import csv
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

CSV_FILE_PATH = 'vehicle_data.csv'

def init_csv_file():
    try:
        with open(CSV_FILE_PATH, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Location Name', 'Vehicle In Count', 'Vehicle Out Count']) 
    except FileExistsError:
        pass

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    location_name = data.get('Location_Name')
    vehicle_in_count = data.get('vehicle_in_count')
    vehicle_out_count = data.get('vehicle_out_count')
    
    # 받은 데이터를 CSV 파일에 누적 저장 (append 모드)
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([location_name, vehicle_in_count, vehicle_out_count])
    
    return jsonify({
        "message": f"Received {location_name} {vehicle_in_count} {vehicle_out_count}"
    })

@app.route('/view_data')
def view_data():
    # CSV 파일에서 데이터를 읽어와 HTML로 렌더링
    with open(CSV_FILE_PATH, mode='r') as file:
        csv_data = list(csv.reader(file))
    
    return render_template('view_data.html', data=csv_data)


if __name__ == '__main__':
    init_csv_file()  # 서버 시작 시 CSV 파일 초기화 (없으면 생성)
    app.run(debug=True, host='0.0.0.0', port=5000)
