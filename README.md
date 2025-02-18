# Drowsiness Detection Using OpenCV and Dlib

This project implements a real-time drowsiness detection system that uses a webcam feed to monitor eye activity to determine if a person is active, drowsy, or asleep. It leverages the `OpenCV` library for image processing, `dlib` for face detection and facial landmark prediction, and `imutils` for handling facial landmarks.

## Code Breakdown

1. **Imports**:
    ```python
    import cv2 as cv
    import numpy as np
    import dlib
    from imutils import face_utils
    ```
    - `cv2` (OpenCV) is used for image processing, capturing webcam input, and displaying results.
    - `numpy` is used for numerical operations, like calculating distances.
    - `dlib` provides tools for facial landmark detection.
    - `face_utils` (from `imutils`) helps convert dlib's landmark predictions into a NumPy array.

2. **Variables and Constants**:
    ```python
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    ```
    - `detector`: The face detector is initialized using `dlib.get_frontal_face_detector()`, which detects faces in an image.
    - `predictor`: This loads the pre-trained shape predictor model that provides 68 facial landmarks.

3. **Tracking variables**:
    ```python
    sleep = 0
    drowsy = 0
    active = 0
    status = ""
    color = (0,0,0)
    ```
    These variables track the state of the user:
    - `sleep`, `drowsy`, and `active`: Counters that track the number of frames for each state.
    - `status`: Stores the current state as a string (`"SLEEPING!"`, `"DROWSY!"`, or `"ACTIVE!"`).
    - `color`: Determines the color to display the status on the screen (blue for sleep, red for drowsy, and green for active).

4. **Distance Function (`dist`)**:
    ```python
    def dist(a, b):
        dst = np.linalg.norm(a-b)
        return dst
    ```
    This function calculates the Euclidean distance between two points `a` and `b`. This distance is later used to calculate the blink ratio.

5. **Blink Detection Function (`blinked`)**:
    ```python
    def blinked(a,b,c,d,e,f):
        up = dist(b, d) + dist(c,e)
        down = dist(a, f)
        ratio = up/(2.0*down)
    
        if ratio > 0.25:
            return 2
        elif 0.25 >= ratio > 0.21:
            return 1
        else:
            return 0
    ```
    - This function checks if a person has blinked or is drowsy based on eye landmarks.
    - It calculates the vertical distance (`up`) between the upper and lower eyelids and divides it by the horizontal distance (`down`), which gives the blink ratio.
    - The function returns:
      - `2` if the blink ratio is greater than 0.25 (eyes are closed).
      - `1` if the blink ratio is between 0.21 and 0.25 (semi-closed eyes, indicating drowsiness).
      - `0` if the blink ratio is less than 0.21 (eyes are open, indicating alertness).

6. **Webcam Capture Loop**:
    ```python
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector(gray)
    ```
    - `cap = cv.VideoCapture(0)` initializes the webcam feed.
    - In the `while True` loop, it reads each frame from the webcam and converts it to grayscale (`gray`) for face detection.

7. **Face Detection**:
    ```python
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_frame = frame.copy()
        cv.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
    ```
    - For each detected face, it draws a green rectangle around the face (`cv.rectangle`).
    - `landmarks = predictor(gray, face)` detects 68 facial landmarks.
    - The landmarks are then converted into a NumPy array for easier manipulation.

8. **Blink Detection for Left and Right Eyes**:
    ```python
    left_blink = blinked(landmarks[36],landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
    right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
    ```
    - The left and right eye landmarks (indices 36-41 and 42-47, respectively) are passed to the `blinked()` function to calculate the blink status for both eyes.

9. **State Classification (Active, Drowsy, or Sleeping)**:
    ```python
    if left_blink == 0 and right_blink == 0:
        sleep += 1
        drowsy = 0
        active = 0
        if sleep > 6:
            status = "SLEEPING!"
            color = (255, 0, 0)
    elif left_blink == 1 and right_blink == 1:
        sleep = 0
        active = 0
        drowsy += 1
        if drowsy > 6:
            status = "DROWSY!"
            color = (0, 0, 255)
    else:
        drowsy = 0
        sleep = 0
        active += 1
        if active > 6:
            status = "ACTIVE!"
            color = (0, 255, 0)
    ```
    - If both eyes are open (blink status `0`), it increments the `sleep` counter.
    - If both eyes show semi-closed status (blink status `1`), it increments the `drowsy` counter.
    - If there is no blink detected (both eyes closed for a moment), it increments the `active` counter.
    - Based on these counters, the user's state is updated to either "SLEEPING!", "DROWSY!", or "ACTIVE!", and the text color is updated accordingly.

10. **Display Status on Screen**:
    ```python
    cv.putText(frame, status, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    for n in range(0, 68):
        (x, y) = landmarks[n]
        cv.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
    ```
    - The `cv.putText()` function displays the current status ("ACTIVE!", "DROWSY!", or "SLEEPING!") on the webcam feed.
    - The landmarks (68 points) are drawn on the face for visual feedback using `cv.circle()`.

11. **Display the Frames**:
    ```python
    cv.imshow("Frame", frame)
    cv.imshow("Result of detector", face_frame)
    ```
    - The `frame` (original with status) and `face_frame` (with landmarks) are shown in two separate windows.

12. **Exit on Keypress**:
    ```python
    key = cv.waitKey(1)
    if key == 27:
        break
    ```
    - The loop continues until the `Esc` key (ASCII 27) is pressed, exiting the program.

## Download the Shape Predictor File

To use the facial landmark detection functionality, you need to download the `shape_predictor_68_face_landmarks.dat` file. You can get it from the following link:

- [Download shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Once downloaded, extract the `.bz2` file and place the `.dat` file in the same directory as the script.
