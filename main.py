from random import random
import cv2
import pygame
import time
import os
from deepface import DeepFace
import random
from multiprocessing import Process, Queue
from FastDeepFace import FastDeepFace

frame_count = 0
start_time = time.time()
fps = 0
audio_path = "audio"
prev_stable_count = 0
MIN_CONFIDENCE = 75
MIN_WIDTH = 80
MIN_FRAME_COUNT = 5


def sound_loop(queue):
    pygame.mixer.init()
    print("[INFO] Sound loop started, waiting for messages...")
    while True:
        msg = queue.get()  # Read from the queue and do nothing
        if msg == "DONE":
            break
        print(f"[INFO] Playing sound: {msg}")
        pygame.mixer.music.load(msg)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() == True:
           continue


def playsound(name):
    filename = random.choice([x for x in os.listdir(os.path.join(audio_path, name)) if not x.startswith(".")])
    q.put(os.path.join(audio_path, name, filename))    

def process_frame(frame):
    # grab_facial_areas
    try:
        dff.extract_faces( img_path=frame)
        dff.filter_by_width(threshold=75)
        dfs = dff.find( )

        faces = [
                (
                    face_obj["facial_area"]["x"],
                    face_obj["facial_area"]["y"],
                    face_obj["facial_area"]["w"],
                    face_obj["facial_area"]["h"],
                    face_obj["facial_area"]["left_eye"],
                    face_obj["facial_area"]["right_eye"],
                    face_obj["embedding"],
                    face_obj["confidence"],
                    face_obj.get("is_real", True),
                    os.path.basename(os.path.dirname(dfs[idx].iloc[0]["identity"])) if dfs[idx].iloc[0]["confidence"] > MIN_CONFIDENCE else f"Unknown{dff.get_next_id()}",
                    dfs[idx].iloc[0]["confidence"],
                    dfs[idx].iloc[0]["gender"],
                    dfs[idx].iloc[0]["last_seen"],
                    dfs[idx].iloc[0]["frame_count"]
                )
                for idx, face_obj in enumerate(dff.source_objs)
                if face_obj["facial_area"]["w"] > MIN_WIDTH
            ]
        dff.append_to_database([x for x in faces if x[-4] <= MIN_CONFIDENCE], frame)
        dff.do_housekeeping(faces)
        
    except ValueError as err:
        if not str(err).startswith("Face could not be detected in numpy array"):
            print(err)
        if str(err).startswith("Length of values "):
            print("Length of values mismatch in database processing.")
        faces = [] # no faces detected

    return faces

def draw_results(frame, faces):
    # Here you would draw boxes and labels on the frame based on recognition results
    # For demonstration, we will just return the original frame

    for idx, (x, y, w, h, left_eye, right_eye, embedding, confidence, is_real,  label, confidence2, gender, last_seen, frame_count) in enumerate(faces):
        # print(label, gender, last_seen, frame_count, left_eye, right_eye, confidence, is_real, confidence2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.rectangle(frame, (x - 1, y ), (x + w + 1, y + 15), (0, 255, 255), cv2.FILLED)
        # {gender} 
        info = f"{label} {confidence2:.2f} {confidence} "
        cv2.putText(frame, info, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 22, 255) if gender == "Woman" else (255, 22, 0), 1)
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

#------------------------------------------------------------------------------
if __name__ == '__main__':
    q = Queue()
    p = Process(target=sound_loop, args=(q,))
    p.daemon = True
    p.start()

    dff = FastDeepFace(threshold=130)
    dff.load_database()


    # Initialize the USB webcam (0 is usually the default camera index)
    filename = "Crowd_walking_forward_NYC_B-Roll.mp4"
    cap = cv2.VideoCapture(0)

    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

    last_seen = {}
    q.put("game-start-6104.mp3")
    while True:

        # Capture a frame from camera
        has_frame, frame = cap.read()
        if not has_frame:
            break
        # Process the frame with the function
        faces = process_frame(frame)
        now = time.time()
        # for idx, (x, y, w, h, left_eye, right_eye, confidence, is_real, gender, name, confidence2) in enumerate(faces):
        #     if name not in last_seen:
        #         last_seen[name] = 0
        #     if now - last_seen[name] > 60:
        #         try:
        #             filename = random.choice([x for x in os.listdir(os.path.join(audio_path, name)) if not x.startswith(".")])
        #             print(f"[INFO] Detected: {name}, playing audio: {filename}, last seen: {time.time() - last_seen[name]:.1f} seconds ago")
        #             playsound(os.path.join("audio", name, filename))
        #         except Exception as e:
        #             print(f"[ERROR] Failed to play audio for {name}: {e}")
        #     last_seen[name] = now   
            
        # Get the text and boxes to be drawn based on the processed frame
        display_frame = draw_results(frame, faces)
        
        new_knowns = [(label, gender) for (x, y, w, h, left_eye, right_eye, embedding, confidence, is_real,  label, confidence2, gender, last_seen, frame_count) in faces if frame_count == MIN_FRAME_COUNT and not label.startswith("Unknown")]
        for name in new_knowns:
            print(f"[INFO] New known detected: {name}")
            playsound(name)
        stable_faces = [(label, gender) for (x, y, w, h, left_eye, right_eye, embedding, confidence, is_real,  label, confidence2, gender, last_seen, frame_count) in faces if frame_count >= MIN_FRAME_COUNT ]
        if len(stable_faces) > prev_stable_count:
            print(f"[INFO] Stable faces count increased: {len(stable_faces)} (Previous: {prev_stable_count})")
            if len(stable_faces) == 1 :
                print(f"[INFO] Single stable face detected: {stable_faces[0][0]} ({stable_faces[0][1]})")
                if stable_faces[0][1]=="Man":
                    playsound("unknown_m")
                else:
                    playsound("unknown_f")
            elif len(stable_faces) == 2:
                playsound("unknown_duo")
            elif len(stable_faces) > 2:
                playsound("unknown_group")
    
        prev_stable_count = len(stable_faces)
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
    q.put("DONE")  # Signal the sound loop to exit
    cap.release()
    cv2.destroyAllWindows()
    DeepFace.stream()
