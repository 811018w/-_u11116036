import cv2
import numpy as np
import joblib

# 載入模型
print('loading...')
knn = joblib.load('mnist_knn.pkl')
print('start...')

# 啟用攝影鏡頭
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow('U11116036', cv2.WINDOW_NORMAL)

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    # 確認影像大小
    img_height, img_width, _ = img.shape

    # 定義擷取區域與處理
    x, y, w, h = 490, 300, 80, 80
    img_num = img[y:y+h, x:x+w]
    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)    # 顏色轉成灰階
    _, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)  # 二值化
    output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)     # 轉回彩色

    # 動態調整顯示區域
    roi_x, roi_y = img_width - w, 0  # 右上角區域開始座標
    if roi_x + w <= img_width and roi_y + h <= img_height:
        img[roi_y:roi_y+h, roi_x:roi_x+w] = output

    # 調整為模型輸入格式
    img_num = cv2.resize(img_num, (28, 28)).astype('float32').reshape(1, -1) / 255

    # 辨識數字
    num = knn.predict(img_num)[0]

    # 顯示辨識結果
    cv2.putText(img, str(int(num)), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 100, 255), 3)
    cv2.imshow('U11116036', img)

    if cv2.waitKey(50) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
