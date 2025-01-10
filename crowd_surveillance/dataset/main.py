import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import geocoder
import csv

# Path where the training images are stored
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Load training images and class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to find the encodings of images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to get the current location of the system
def get_location():
    g = geocoder.ip('me')  # Get the location using the IP address
    return g.city, g.country  # Return city and country

# Function to read and write criminal records with time and location
def markcriminal_record(name, place, last_detected):
    current_time = datetime.now()

    # Convert the current time to 12-hour format with AM/PM
    dtString = current_time.strftime('%I:%M:%S %p')  # 12-hour format with AM/PM
    dateString = current_time.strftime('%Y-%m-%d')  # Current date

    try:
        with open('criminal_record.csv', 'r+', newline='') as f:
            reader = csv.reader(f)
            existing_data = list(reader)

            # Check if the name already exists in the CSV
            for row in existing_data:
                if row[0] == name:
                    last_entry_time = datetime.strptime(row[1], '%I:%M:%S %p')  # 12-hour format
                    time_diff = current_time - last_entry_time
                    if time_diff >= timedelta(minutes=5):  # Check if 5 minutes have passed
                        writer = csv.writer(f)
                        writer.writerow([name, dtString, dateString, place])
                        last_detected[name] = current_time  # Update last detected time
                        break
            else:
                # If the name doesn't exist in the CSV, add a new entry
                with open('criminal_record.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, dtString, dateString, place])
                last_detected[name] = current_time  # Store the current time for new entry
    
    except FileNotFoundError:
        # If the CSV file doesn't exist, create a new one
        with open('criminal_record.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Time', 'Date', 'Place'])  # Write header
            writer.writerow([name, dtString, dateString, place])
        last_detected[name] = current_time  # Store the current time for the first detected entry


# Load encodings for known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get location (city and country)
place = get_location()  # This will return a tuple (city, country)

# Dictionary to store the last detected time of each person
last_detected = {}

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Check if 5 minutes have passed since last entry and add the new entry
            if name not in last_detected or datetime.now() - last_detected[name] >= timedelta(minutes=5):
                markcriminal_record(name, place, last_detected)

    # Display the webcam feed with rectangles around faces
    cv2.imshow('Webcam', img)

    # Wait for key press, if 'Q' is pressed, exit the loop and release the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
