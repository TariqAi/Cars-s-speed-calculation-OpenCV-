import cv2
import time

cascade_src = 'cars.xml'
video_src = 'data/video3.MP4'

ax1, ay, ax2 = 70, 90, 230
bx1, by, bx2 = 15, 125, 225
distance = 9.144  # Distance between lines 'a' and 'b' in meters

def Speed_Cal(time):
    if time > 0:
        speed_mps = distance / time  # Speed in meters per second
        speed_kmph = speed_mps * 3.6  # Convert to kilometers per hour
        return speed_kmph
    return 0

i = 1
start_time = 0
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

car_detected_a = False
car_detected_b = False

# Get the FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)  # Calculate delay to match video FPS

while True:
    ret, img = cap.read()
    if type(img) == type(None):
        break

    blurred = cv2.blur(img, ksize=(15, 15))
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    cv2.line(img, (ax1, ay), (ax2, ay), (255, 0, 0), 2)
    cv2.line(img, (bx1, by), (bx2, by), (255, 0, 0), 2)

    for (x, y, w, h) in cars:

        cv2.rectangle(img, (x, y), (x + w , y + h), (0, 255, 0), 1)
        #cv2.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), 2, (0, 0, 255), -1)

        center_y = int((y + y + h) / 2)

        if not car_detected_a and center_y >= ay - 5 and center_y <= ay + 5:
            car_detected_a = True
            start_time = time.time()
            cv2.line(img, (ax1, ay), (ax2, ay), (0, 255, 0), 2)

        if car_detected_a and not car_detected_b and center_y >= by - 5 and center_y <= by + 5:
            car_detected_b = True
            end_time = time.time()
            time_diff = end_time - start_time
            speed = Speed_Cal(time_diff)
            print(f"Car Number {i} Speed: {speed:.2f} KM/H")
            i += 1
            car_detected_a = False
            car_detected_b = False

    cv2.imshow('video', img)
    if cv2.waitKey(delay) == 27:  # Use the calculated delay to match FPS
        break

cap.release()
cv2.destroyAllWindows()
