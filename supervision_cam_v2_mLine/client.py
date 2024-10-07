import requests
import json

from data.count import in_count,out_count

# 차량 감지 및 카운팅 후 서버에 보낼 데이터 준비
locationName = "asdf"
vehicle_in_count = 10#in_count
vehicle_out_count = 10#out_count# 예시로 감지된 차량 수

# 서버에 전달할 데이터
data = {
    "Location_Name": locationName,
    "vehicle_in_count": vehicle_in_count,
    'vehicle_out_count': vehicle_out_count
}

# 서버에 POST 요청 보내기
response = requests.post('http://203.250.35.96:5000/receive_data', json=data)

# 서버 응답 출력
print(response.json())
