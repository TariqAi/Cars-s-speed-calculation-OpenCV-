import cv2
import time
import numpy as np

# Paths to cascade classifier and video file
cascade_src = 'cars.xml'
video_src = 'data/video3.MP4'

# Define line coordinates and distance between lines 'a' and 'b' in meters
ax1, ay, ax2 = 70, 90, 230
bx1, by, bx2 = 15, 125, 225
distance = 9.144

def Speed_Cal(time):
    # Calculate speed in km/h based on time in seconds
    if time > 0:
        speed_mps = distance / time  # Speed in meters per second
        speed_kmph = speed_mps * 3.6  # Convert to kilometers per hour
        return speed_kmph
    return 0

i = 1
start_time = 0
cap = cv2.VideoCapture(video_src)  # Open video file
car_cascade = cv2.CascadeClassifier(cascade_src)  # Load car detector

car_detected_a = False
car_detected_b = False

# Get FPS from video to match display delay
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)  # Delay in milliseconds

# Define ROI coordinates (top-left and bottom-right)
roi_x1, roi_y1 = 30, 50
roi_x2, roi_y2 = 240, 150

while True:
    ret, img = cap.read()  # Read frame from video
    if not ret:
        break

    # Convert to grayscale and apply Gaussian blur for better thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)  # Reduce noise
    
    # Apply binary thresholding to create mask
    _, mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Create a mask for the ROI
    roi_mask = np.zeros_like(gray)
    roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255

    # Apply ROI mask to the thresholded mask
    masked_mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    # Apply ROI mask to the original image
    masked_img = cv2.bitwise_and(img, img, mask=roi_mask)

    # Detect cars in the blurred image
    cars = car_cascade.detectMultiScale(blurred, 1.1, 2)

    # Draw reference lines on the original image
    cv2.line(img, (ax1, ay), (ax2, ay), (255, 0, 0), 2)
    cv2.line(img, (bx1, by), (bx2, by), (255, 0, 0), 2)

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        center_y = int((y + y + h) / 2)

        # Check if car crosses line 'a'
        if not car_detected_a and center_y >= ay - 5 and center_y <= ay + 5:
            car_detected_a = True
            start_time = time.time()
            cv2.line(img, (ax1, ay), (ax2, ay), (0, 255, 0), 2)

        # Check if car crosses line 'b'
        if car_detected_a and not car_detected_b and center_y >= by - 5 and center_y <= by + 5:
            car_detected_b = True
            end_time = time.time()
            time_diff = end_time - start_time
            speed = Speed_Cal(time_diff)
            print(f"Car Number {i} Speed: {speed:.2f} KM/H")
            i += 1
            car_detected_a = False
            car_detected_b = False

    # Display the original frame with detections
    cv2.imshow('video', img)
    
    # Display the thresholded mask within ROI
    cv2.imshow('mask', masked_mask)
    
    # Display the original frame within ROI
    cv2.imshow('color_mask', masked_img)

    # Break the loop if 'ESC' key is pressed
    if cv2.waitKey(delay) == 27:
        break

cap.release()  # Release video capture
cv2.destroyAllWindows()  # Close all OpenCV windows
