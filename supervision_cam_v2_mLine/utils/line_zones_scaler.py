def scale_line_zones(line_zone_points, input_width, input_height, base_width=1920, base_height=1080):
    """입력 해상도에 맞게 라인 존 좌표를 스케일링하는 함수"""
    scale_x = input_width / base_width
    scale_y = input_height / base_height
    scaled_line_zones = []

    for start_x, start_y, end_x, end_y in line_zone_points:
        scaled_start_x = int(start_x * scale_x)
        scaled_start_y = int(start_y * scale_y)
        scaled_end_x = int(end_x * scale_x)
        scaled_end_y = int(end_y * scale_y)
        
        scaled_line_zones.append((scaled_start_x, scaled_start_y, scaled_end_x, scaled_end_y))

    return scaled_line_zones