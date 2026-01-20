import cv2
import pygame
import time
import os
from deepface import DeepFace

threshold: int = 130
anti_spoofing: bool = False
db_path: str = "database"
model_name: str = "VGG-Face"
detector_backend: str = "opencv"
distance_metric: str = "cosine"

pygame.mixer.init()

DeepFace.build_model(task="facial_attribute", model_name="Age")
DeepFace.build_model(task="facial_attribute", model_name="Gender")
DeepFace.build_model(task="facial_attribute", model_name="Emotion")
DeepFace.build_model(task="facial_recognition", model_name=model_name)

# Initialize the USB webcam (0 is usually the default camera index)
cap = cv2.VideoCapture(0)

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
cv_scaler = 4 # this has to be a whole number
frame_count = 0
start_time = time.time()
fps = 0

def playsound(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    #while pygame.mixer.music.get_busy() == True:
    #    continue

def process_frame(frame):
    # grab_facial_areas
    try:
        face_objs = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=detector_backend,
                    # you may consider to extract with larger expanding value
                    expand_percentage=0,
                    anti_spoofing=anti_spoofing,
                )
        faces = [
                (
                    face_obj["facial_area"]["x"],
                    face_obj["facial_area"]["y"],
                    face_obj["facial_area"]["w"],
                    face_obj["facial_area"]["h"],
                    face_obj["facial_area"]["left_eye"],
                    face_obj["facial_area"]["right_eye"],
                    face_obj["confidence"],
                    face_obj.get("is_real", True),
                    face_obj.get("antispoof_score", 0),
                )
                for face_obj in face_objs
                if face_obj["facial_area"]["w"] > threshold
            ]
    except Exception as e:
        faces = []

    for idx, (x, y, w, h, left_eye, right_eye, confidence, is_real, antispoof_score) in enumerate(faces):
        detected_face = frame[int(y) : int(y + h), int(x) : int(x + w)]

        # perform_demography_analysis
        demographies =  DeepFace.analyze(
                img_path=detected_face,
                actions=("age", "gender", "emotion"),
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
        faces[idx] += (demographies[0]["age"], demographies[0]["dominant_gender"], demographies[0]["dominant_emotion"])

        # perform_facial_recognition
        label = None
        confidence = 0
        try:
            dfs = DeepFace.find(
                img_path=detected_face,
                db_path=db_path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False,
                silent=True,
            )
        except ValueError as err:
            print("Face not found in database:", err)
            dfs = []
        if len(dfs) > 0 and len(dfs[0]) > 0:
            best_match = dfs[0].iloc[0]
            label = os.path.basename(os.path.dirname(best_match["identity"]))
            confidence = best_match["confidence"]
            faces[idx] += (label, confidence)
        else:
            faces[idx] += ('Unknown', 0)  
    return faces

def draw_results(frame, faces):
    # Here you would draw boxes and labels on the frame based on recognition results
    # For demonstration, we will just return the original frame
    for idx, (x, y, w, h, left_eye, right_eye, confidence, is_real, antispoof_score, age, gender, emotion, label, confidence2) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.rectangle(frame, (x - 1, y - 35), (x+w+1, y), (0, 255, 255), cv2.FILLED)
        info1 = f"{gender} {age}, {emotion}"
        info2 = f"conf {int(100*confidence)}% score {int(confidence2)}%"
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(frame, info1, (x+50, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(frame, info2, (x+50, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        if left_eye:
            cv2.circle(frame, (left_eye[0], left_eye[1]), 5, (255, 0, 0), -1)
        if right_eye:
            cv2.circle(frame, (right_eye[0], right_eye[1]), 5, (255, 0, 0), -1)
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


while True:
    # Capture a frame from camera
    has_frame, frame = cap.read()
    if not has_frame:
        break
    # Process the frame with the function
    objs = process_frame(frame)
    
    # for name in face_names:
    #     if name not in last_seen:
    #         last_seen[name] = 0
    #     if time.time() - last_seen[name] > 60:
    #         filename = random.choice([x for x in os.listdir(os.path.join("audio$
    #         print(f"[INFO] Detected: {name} playing audio: {filename}")
    #         playsound(os.path.join("audio", name, filename))
    #     last_seen[name] = time.time()   
        
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(frame, faces=objs)
    
    # Calculate and update FPS
    current_fps = calculate_fps()
    
    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display everything over the video feed.
    cv2.imshow('Face Rec Running', display_frame)
        
    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("a"):
        print("Saving capture.jpg")
        cv2.imwrite("capture.jpg", display_frame)

# By breaking the loop we run this code here which closes everything
cap.release()
cv2.destroyAllWindows()
    
