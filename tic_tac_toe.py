import cv2
import numpy as np
import math

def check_state(filename: str) -> str:
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "Ongoing"
    board_cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(board_cnt, True)
    approx = cv2.approxPolyDP(board_cnt, 0.02 * peri, True)
    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(board_cnt)
        approx = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
    pts = np.array([pt[0] for pt in approx], dtype="float32")
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    board_size = 300
    dst = np.array([[0, 0], [board_size - 1, 0], [board_size - 1, board_size - 1], [0, board_size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (board_size, board_size))
    cell_size = board_size // 3
    board = [["" for _ in range(3)] for _ in range(3)]
    def detect_symbol(cell):
        m = int(cell.shape[0] * 0.12)
        if cell.shape[0] - 2 * m <= 0 or cell.shape[1] - 2 * m <= 0:
            return ""
        roi = cell[m:cell.shape[0]-m, m:cell.shape[1]-m]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_th = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        proc = cv2.morphologyEx(roi_th, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        proc = cv2.dilate(proc, kernel_small, iterations=2)
        cnts_roi, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_roi:
            return ""
        cmax = max(cnts_roi, key=cv2.contourArea)
        area = cv2.contourArea(cmax)
        if area < (roi.shape[0] * roi.shape[1] * 0.02):
            return ""
        perim = cv2.arcLength(cmax, True)
        if perim == 0:
            return ""
        circularity = 4 * math.pi * area / (perim * perim)
        x, y, w, h = cv2.boundingRect(cmax)
        rect_ratio = area / ((w * h) + 1e-5)
        approx_poly = cv2.approxPolyDP(cmax, 0.02 * perim, True)
        if circularity > 0.72 and rect_ratio > 0.65:
            return "O"
        elif circularity < 0.60:
            return "X"
        else:
            return "X" if len(approx_poly) >= 6 else "O"
    for i in range(3):
        for j in range(3):
            x = j * cell_size
            y = i * cell_size
            cell = warped[y:y+cell_size, x:x+cell_size]
            board[i][j] = detect_symbol(cell)
    def check_winner(b, p):
        for i in range(3):
            if all(b[i][j] == p for j in range(3)):
                return True
        for j in range(3):
            if all(b[i][j] == p for i in range(3)):
                return True
        if b[0][0] == p and b[1][1] == p and b[2][2] == p:
            return True
        if b[0][2] == p and b[1][1] == p and b[2][0] == p:
            return True
        return False
    if check_winner(board, "X"):
        return "X Wins"
    if check_winner(board, "O"):
        return "O Wins"
    filled = sum(1 for i in range(3) for j in range(3) if board[i][j] != "")
    if filled == 9:
        return "Draw"
    return "Ongoing"

# if __name__ == "__main__":
#     test_file = "xo.png"
#     result = check_state(test_file)
#     print("Detected State:", result)
