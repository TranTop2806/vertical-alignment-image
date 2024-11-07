import cv2
import numpy as np
import os

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_suffix_from_filename(filename):
    return "_" + os.path.splitext(os.path.basename(filename))[0].split("input")[-1]

def detect_main_vertical_edges(image, suffix):
    # Chuyển ảnh sang xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh để giảm nhiễu 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện cạnh 
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    cv2.imwrite(os.path.join(output_dir, f'edges_canny_blurred{suffix}.jpg'), edges)

    # Phát hiện các đường thẳng chính bằng Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    
    # Lọc các đường dọc chính 
    vertical_lines = []
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                angle = np.degrees(theta)
                if -15 <= angle <= 15 or 165 <= angle <= 195:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    vertical_lines.append((rho, theta))
    
    cv2.imwrite(os.path.join(output_dir, f'detected_main_edges{suffix}.jpg'), image)
    
    return vertical_lines

# Tính toán góc lệch 
def calculate_alignment_angle(vertical_lines):
    if not vertical_lines:
        print("Không phát hiện được cạnh chính nào.")
        return 0  
    
    angles = []
    for rho, theta in vertical_lines:
        angle = np.degrees(theta)
        deviation = angle if angle <= 90 else angle - 180
        angles.append(deviation)
    
    # Tính góc lệch trung bình
    if angles:
        average_deviation = np.mean(angles)
        print(f"Góc lệch trung bình từ trục đứng: {average_deviation:.2f} độ")
        return average_deviation
    else:
        print("Không tìm thấy cạnh chính gần vuông góc với trục hoành.")
    return 0

# Xoay ảnh 
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Hàm căn chỉnh ảnh
def align_image_vertically(image_path, outputname):
    suffix = get_suffix_from_filename(image_path)
    
    image = cv2.imread(image_path)
    if image is None:
        print("Lỗi: Không tìm thấy ảnh.")
        return
    
    vertical_lines = detect_main_vertical_edges(image.copy(), suffix)
    angle = calculate_alignment_angle(vertical_lines)
    aligned_image = rotate_image(image, angle)
    output_filename = os.path.join(output_dir, f"{outputname.split('.')[0]}{suffix}.jpg")
    cv2.imwrite(output_filename, aligned_image)
    print(f"Đã lưu ảnh đã căn chỉnh tại {output_filename}")

if __name__ == '__main__':
    output_dir = 'data/output'
    create_dir_if_not_exists(output_dir)
    align_image_vertically('data/input/input03.jpg', 'aligned_output.jpg')
