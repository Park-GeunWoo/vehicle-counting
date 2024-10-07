# data/data_store.py
in_count = [0]
out_count = [0]

import json
JSON_FILE_PATH = 'data_store.json'

data_store = {
    "location_name": "Unknown",
    "ioo": None,
    "vid_info": None,
    "total_frames": 0,  # 총 프레임 수
    "avg_fps": 0,  # 평균 FPS
    "total_time": 0  # 총 시간
}

# 데이터를 JSON 파일에 저장
def save_data():
    with open(JSON_FILE_PATH, 'w') as json_file:
        json.dump({
            "data_store": data_store,
            "in_count": in_count[0],
            "out_count": out_count[0]
        }, json_file, indent=4)
    
    
def update(new_name, send_val,video_info,total_frames,avg_fps,total_time):
    global data_store,in_count,out_count
    data_store["location_name"] = new_name  # 딕셔너리의 location_name 업데이트
    data_store["ioo"] = send_val  # 딕셔너리의 ioo 업데이트
    data_store["vid_info"] = video_info  # 딕셔너리의 vid_info 업데이트
    data_store['total_frames']=total_frames
    data_store['avg_fps']=avg_fps
    data_store['total_frames']=total_time
    save_data()