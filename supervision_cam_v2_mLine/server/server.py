import csv
import logging
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

CSV_FILE_PATH = 'vehicle_data.csv'

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def init_csv_file():
    try:
        with open(CSV_FILE_PATH, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Edge ID','Location', 'GPS' ,'Time', 'Count'])
    except FileExistsError:
        pass


@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    print(data)
    count = data.get('count')
    data_store = data.get('data_store', {})
    edge_id=data_store.get('edge_id')
    location_name = data_store.get('location_name')
    gps_point=data_store.get('gps')
    time = data_store.get('time')

    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([edge_id,location_name,gps_point,time,count])
    return jsonify({
        "message": f"ID {edge_id}, Received {location_name}, GPS {gps_point}, Count {count}, Time {time}"
    })

@app.route('/view_data')
def view_data():
    
    with open(CSV_FILE_PATH, mode='r') as file:
        csv_data = list(csv.reader(file))
    
    return render_template('view_data.html', data=csv_data)


if __name__ == '__main__':
    init_csv_file()
    app.run(debug=True, host='0.0.0.0', port=5000)
    #rstp
