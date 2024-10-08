import csv
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

CSV_FILE_PATH = 'vehicle_data.csv'

def init_csv_file():
    try:
        with open(CSV_FILE_PATH, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Location Name', 'In Count', 'Out Count', 
                             'Video Info', 'Total Time', 'Average FPS', 'Total Frames'])
    except FileExistsError:
        pass


@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    
    vehicle_in_count = data.get('in_count')
    vehicle_out_count = data.get('out_count')
    
    data_store = data.get('data_store', {})
    location_name = data_store.get('location_name')
    vid_info = data_store.get('vid_info')
    total_time = data_store.get('total_time')
    avg_fps = data_store.get('avg_fps')
    total_frames = data_store.get('total_frames')

    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([location_name, vehicle_in_count, vehicle_out_count, 
                         vid_info, total_time, avg_fps, total_frames])
    
    return jsonify({
        "message": f"Received {location_name}, In: {vehicle_in_count}, Out: {vehicle_out_count}, Video Info: {vid_info}, Total Time: {total_time}, Average FPS: {avg_fps}, Total Frames: {total_frames}"
    })

@app.route('/view_data')
def view_data():
    
    with open(CSV_FILE_PATH, mode='r') as file:
        csv_data = list(csv.reader(file))
    
    return render_template('view_data.html', data=csv_data)


if __name__ == '__main__':
    init_csv_file()
    app.run(debug=True, host='0.0.0.0', port=5000)
