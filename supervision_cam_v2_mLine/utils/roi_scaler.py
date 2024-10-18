def scale_roi(roi_points, input_width, input_height, base_width=1920, base_height=1080):
    """해상도에 맞춰 ROI 스케일링하는 함수"""
    scale_x = input_width / base_width
    scale_y = input_height / base_height
    x_min = int(roi_points[0] * scale_x)
    y_min = int(roi_points[1] * scale_y)
    x_max = int(roi_points[2] * scale_x)
    y_max = int(roi_points[3] * scale_y)
    return (x_min, y_min, x_max, y_max)