from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from app.verify import authentication
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
from datetime import datetime
import cv2
from time import time
# import mediapipe as mp
from .process import *
from django.shortcuts import render,redirect, get_object_or_404


from .models import Notification
# Create your views here.
def index(request):
    # return HttpResponse("This is Home page")    
    return render(request, "index.html")


def admin_log_in(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Hardcoded username and password
        if username == "user@gmail.com" and password == "user":
            messages.success(request, "Admin Log In Successful...!")
            return redirect("admin_dashboard")  # Redirect to admin dashboard or any appropriate page
        else:
            messages.error(request, "Invalid Admin Credentials...!")
            return redirect("admin_log_in")
    return render(request, "admin_log_in.html")

from django.utils import timezone
from datetime import timedelta
from django.utils.timezone import now
def admin_dashboard(request):
    # Fetch notifications created within the last 5 minutes
        time_threshold = now() - timedelta(minutes=2)
        notifications = Notification.objects.filter(created_at__gte=time_threshold)

        users = User.objects.all()  # Fetch all users from the database
        return render(request, "admin_dashboard.html", {'users': users, 'notifications': notifications})

def delete_user(request, user_id):
    # Get the user by ID, excluding superuser or admin if needed
    user = get_object_or_404(User, id=user_id)
    
    # Ensure only certain users can be deleted
    if not user.is_superuser and user.username not in ["admin", "user"]:
        user.delete()
        messages.success(request, 'User deleted successfully.')
    else:
        messages.error(request, 'This user cannot be deleted.')
    
    # Redirect to the user records page (replace with the actual view)
    return redirect('admindashboard')  # Replace 'admindashboard' with the correct URL name

def log_in(request):
    if request.method == "POST":
        # return HttpResponse("This is Home page")  
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username = username, password = password)

        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    # return HttpResponse("This is Home page")    
    return render(request, "log_in.html")

def register(request):
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['password']
        password1 = request.POST['password1']
        # print(fname, contact_no, ussername)
        verify = authentication(fname, lname, password, password1)
        if verify == "success":
            user = User.objects.create_user(username, password, password1)          #create_user
            user.first_name = fname
            user.last_name = lname
            user.save()
            messages.success(request, "Your Account has been Created.")
            return redirect("/")
            
        else:
            messages.error(request, verify)
            return redirect("register")
    # return HttpResponse("This is Home page")    
    return render(request, "register.html")


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfuly...!")
    return redirect("/")


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import geocoder
import csv
import threading
import math
import cvzone
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
from .models import Notification
from ultralytics import YOLO

# Path where the training images are stored
path = 'dataset/Training_images'
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
    dtString = current_time.strftime('%I:%M:%S %p')  # 12-hour format with AM/PM
    dateString = current_time.strftime('%Y-%m-%d')  # Current date

    try:
        with open('dataset/criminal_record.csv', 'r+', newline='') as f:
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
                with open('dataset/criminal_record.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, dtString, dateString, place])
                last_detected[name] = current_time  # Store the current time for new entry
    except FileNotFoundError:
        with open('dataset/criminal_record.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Time', 'Date', 'Place'])  # Write header
            writer.writerow([name, dtString, dateString, place])
        last_detected[name] = current_time  # Store the current time for the first detected entry

# Load encodings for known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

@login_required(login_url="log_in")
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def dashboard(request):
    context = {
        'fname': request.user.first_name,
    }

    # Initialize Alarm_Status if it's not already initialized in the session
    if 'Alarm_Status' not in request.session:
        request.session['Alarm_Status'] = False

    Alarm_Status = request.session['Alarm_Status']

    if request.method == "POST":
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Load your YOLO fire detection model
        fire_model = YOLO('dataset/fire.pt')  # Ensure 'fire.pt' is the correct path to your fire detection model
        fire_classnames = ['fire']

        # Dictionary to store previous positions of detected persons
        person_positions = {}
        # Falling detection threshold (adjustable)
        falling_threshold = 50  # Adjust based on sensitivity

        # Initialize variables for criminal detection
        last_detected = {}
        place = get_location()  # This will return a tuple (city, country)

        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame for consistent processing
            frame = cv2.resize(frame, (640, 480))

            # Fire detection using YOLO
            fire_detected = False
            fire_results = fire_model(frame, stream=True)

            # Parse YOLO fire detection results
            for info in fire_results:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])

                    if confidence > 70:  # Fire detection threshold
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cvzone.putTextRect(frame, f'{fire_classnames[Class]} {confidence}%', [x1 + 8, y1 + 30], scale=1.5, thickness=2)
                        fire_detected = True

            # Fire alarm logic and notification
            if fire_detected:
                # Create notification with user's full name
                user_full_name = f"{request.user.first_name} {request.user.last_name}"
                message = f"{user_full_name}'s system detecting fire."
                Notification.objects.create(user=request.user, message=message)

                if not Alarm_Status:
                    threading.Thread(target=play_alarm_sound_function).start()
                    Alarm_Status = True
            else:
                Alarm_Status = False

            # Store Alarm_Status back in session to persist its value across requests
            request.session['Alarm_Status'] = Alarm_Status

            # Person detection (YOLO)
            height, width = frame.shape[:2]
            blob_person = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net_person.setInput(blob_person)
            layer_names_person = net_person.getUnconnectedOutLayersNames()
            outputs_person = net_person.forward(layer_names_person)

            # Initialize lists for bounding boxes, confidences, and class IDs
            boxes_person = []
            confidences_person = []
            class_ids_person = []

            # Minimum confidence threshold for person detection
            conf_threshold_person = 0.5

            # Loop over each output layer
            for output_person in outputs_person:
                for detection_person in output_person:
                    scores_person = detection_person[5:]
                    class_id_person = np.argmax(scores_person)
                    confidence_person = scores_person[class_id_person]

                    if confidence_person > conf_threshold_person:
                        if classes[class_id_person] == "person":  # Check if the detected object is a person
                            box_person = detection_person[0:4] * np.array([width, height, width, height])
                            (center_x_person, center_y_person, box_width_person, box_height_person) = box_person.astype("int")
                            x_person = int(center_x_person - (box_width_person / 2))
                            y_person = int(center_y_person - (box_height_person / 2))

                            boxes_person.append([x_person, y_person, int(box_width_person), int(box_height_person)])
                            confidences_person.append(float(confidence_person))
                            class_ids_person.append(class_id_person)

            # Apply non-maximum suppression to remove overlapping boxes
            nms_threshold_person = 0.3
            indices_person = cv2.dnn.NMSBoxes(boxes_person, confidences_person, conf_threshold_person, nms_threshold_person)

            # Person counting
            person_count = len(indices_person) if len(indices_person) > 0 else 0

            # Display the person count in the top-left corner
            cv2.rectangle(frame, (0, 0), (700, 50), (0, 0, 0), -1)  # Black background
            cv2.putText(frame, f"Persons: {person_count}", (450, 23), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)  # Green text

            if person_count > 2:
                cv2.putText(frame, f"Crowded Area!!", (450, 43), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)  # Red text

            # Loop over the indices for person counting and falling person detection
            if len(indices_person) > 0:
                for i_person in indices_person.flatten():
                    box_person = boxes_person[i_person]
                    x_person, y_person, w_person, h_person = box_person
                    label_person = str(classes[class_ids_person[i_person]])
                    confidence_person = confidences_person[i_person]
                    color_person = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (x_person, y_person), (x_person + w_person, y_person + h_person), color_person, 2)
                    text_person = f"{label_person}: {confidence_person:.2f}"
                    cv2.putText(frame, text_person, (x_person, y_person - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_person, 2)

                    # Falling detection logic
                    person_id = i_person  # Use the index as a temporary identifier
                    current_position = (x_person, y_person, w_person, h_person)

                    if person_id in person_positions:
                        prev_position = person_positions[person_id]
                        # Compare current and previous positions for falling detection
                        if abs(current_position[1] - prev_position[1]) > falling_threshold and current_position[3] < prev_position[3]:
                            # Detected a falling motion
                            cv2.putText(frame, "Fall Detected!", (x_person, y_person - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # Create notification for fall detection
                            user_full_name = f"{request.user.first_name} {request.user.last_name}"
                            message = f"{user_full_name}'s system detected a fall. Please check the area!"
                            Notification.objects.create(user=request.user, message=message)

                    # Update position for the current person
                    person_positions[person_id] = current_position

                    # Criminal Identification
                    imgS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                            # Check if 5 minutes have passed since last entry and add the new entry
                            if name not in last_detected or datetime.now() - last_detected[name] >= timedelta(minutes=5):
                                markcriminal_record(name, place, last_detected)

            # Display the frame with object detection
            cv2.imshow("Combined Detection", frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

    return render(request, "dashboard.html", context)


import requests

# def fire_detected():
#     # Notify the server
#     requests.post("http://http://127.0.0.1:8000//notify", json={"event": "fire_detected"})

# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json

# @csrf_exempt
# def fire_notify(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         if data.get('event') == 'fire_detected':
#             # Save notification to the database or send a signal
#             print("Fire detected!")
#             return JsonResponse({"status": "success"})
#     return JsonResponse({"status": "invalid request"})

# from django.http import JsonResponse

# def check_notifications(request):
#     # Replace this with real notification checking logic
#     fire_detected = True  # This should be dynamically checked
#     return JsonResponse({'fire_detected': fire_detected})

