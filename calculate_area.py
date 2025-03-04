import cv2
import numpy as np

def calculate_area(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_bounds = {
        "red": [((0, 100, 100), (10, 255, 255)),
                ((160, 100, 100), (180, 255, 255))],
        "green": [((40, 100, 100), (80, 255, 255))],
        "blue": [((100, 150, 0), (130, 255, 255))],
        "yellow": [((20, 100, 100), (30, 255, 255))],
        "purple": [((140, 50, 50), (170, 255, 255))],
        "gray": [((0, 0, 50), (180, 50, 200))]
    }
    area_dict = {}

    kernel = np.ones((3, 3), np.uint8)

    for color_name in color_bounds:
        masks = []
        for lower, upper in color_bounds[color_name]:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            m = cv2.inRange(hsv, lower, upper)
            masks.append(m)
        mask = masks[0] if len(masks) == 1 else cv2.bitwise_or(masks[0], masks[1])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = 0
        for contour in contours:
    
            if cv2.contourArea(contour) < 10:
                continue

            area = cv2.contourArea(contour)
  
            rect = cv2.minAreaRect(contour)
            (center, (w, h), angle) = rect

            if min(w, h) < 1e-5:
                continue
            ratio = max(w, h) / min(w, h)

            if ratio <= 1.1:
                shape_area = area
            else:
                shape_area = 3 * area

            total_area += shape_area

        if total_area > 0:
            area_dict[color_name] = int(round(total_area))

    if not area_dict:
        total_pixels = image.shape[0] * image.shape[1]
        return f"black, {total_pixels}"

    color_order = ["red", "green", "blue", "yellow", "purple", "gray"]
    result_lines = []
    for color in color_order:
        if color in area_dict:
            result_lines.append(f"{color}, {area_dict[color]}")
    result = "\n".join(result_lines)
    return result


# if __name__ == "__main__":
 
#     test_image = cv2.imread("/mnt/data/contest2.png")
#     output = calculate_area(test_image)
#     print(output)
