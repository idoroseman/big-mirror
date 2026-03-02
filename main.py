from random import random
import cv2
import pygame
import time
import datetime
import os
import random
from multiprocessing import Process, Queue
from FastDeepFace import FastDeepFace
import numpy as np

frame_count = 0
start_time = time.time()
fps = 0
audio_path = "audio"
prev_stable_count = 0
now_playing = None

MIN_CONFIDENCE = 55
MIN_WIDTH = 80
MIN_FRAME_COUNT = 15
LAST_SEEN_TIMEOUT = 30

def sound_loop(queue):
    global now_playing
    pygame.mixer.init()
    print("[SOUND] Sound loop started, waiting for messages...")
    while True:
        msg = queue.get()  # Read from the queue and do nothing
        if msg == "DONE":
            break
        print(f"[SOUND] Playing sound: {msg}")
        now_playing = msg
        try:
            pygame.mixer.music.load(msg)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                continue
        except Exception as e:
          pass
        now_playing = None
    print("[SOUND] Sound loop exiting...")


def playsound(name):
    filename = random.choice([x for x in os.listdir(os.path.join(audio_path, name)) if not x.startswith(".")])
    q.put(os.path.join(audio_path, name, filename))    

def process_frame(frame):
    # grab_facial_areas
    try:
        dff.extract_faces( img_path=frame)
        dff.filter_by_width(threshold=75)
        dfaces = dff.find( )
        dfaces = dff.append_new_to_database(dfaces, MIN_CONFIDENCE)
        dff.do_housekeeping(dfaces, LAST_SEEN_TIMEOUT)


    except ValueError as err:
        dfaces = []
        dff.source_objs = []
        if not str(err).startswith("Face could not be detected in numpy array"):
            print(err)
        if str(err).startswith("Length of values "):
            print("Length of values mismatch in database processing.")
    return dfaces

def draw_results(orig, dfaces=None):
    # Here you would draw boxes and labels on the frame based on recognition results
    # For demonstration, we will just return the original frame
    global now_playing
    frame = orig.copy()
    now = time.time()

    cv2.putText(frame, f"{len(dfaces)} identities", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 0, 255), 1)
    
    if now_playing:
        cv2.putText(frame, f"Playing: {os.path.basename(now_playing)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 0, 255), 1)
    for idx, face_obj in enumerate(dff.source_objs):
        try:
            data_frame = dfaces[idx].iloc[0]
            label = data_frame["identity"].split("/")[0] if data_frame["identity"].startswith("Unknown") else data_frame["identity"].split("/")[1]
            x, y, w, h = face_obj["facial_area"]["x"], face_obj["facial_area"]["y"], face_obj["facial_area"]["w"], face_obj["facial_area"]["h"]
            left_eye = None if np.isnan(data_frame["left_eye"]) else data_frame["left_eye"]
            right_eye = None if np.isnan(data_frame["right_eye"]) else data_frame["right_eye"]
            gender = dff.faces[label]['gender'] if label in dff.faces else "Unknown"
            last_seen = dff.faces[label]['last_seen'] if label in dff.faces else now
            frame_count = dff.faces[label]['frame_count'] if label in dff.faces else 0
            confidence = data_frame["confidence"] if "confidence" in data_frame else 0
            # print(label, gender, last_seen, frame_count, left_eye, right_eye, confidence, is_real, confidence2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.rectangle(frame, (x - 1, y ), (x + w + 1, y + 15), (0, 255, 255), cv2.FILLED)
            # {gender} 
            info = f"{label} {confidence:.2f} {(now-last_seen):.1f}s {frame_count} "
            color = (0, 0, 0)
            if  gender == "Woman":
                color = (0, 22, 255)
            elif gender == "Man":
                color = (255, 22, 0)
            cv2.putText(frame, info, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if left_eye:
                cv2.circle(frame, (left_eye[0], left_eye[1]), 5, (255, 0, 0), -1)
            if right_eye:
                cv2.circle(frame, (right_eye[0], right_eye[1]), 5, (255, 0, 0), -1)
        except:
            pass
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

    dff = FastDeepFace()
    dff.load_database()
    print(f"[INFO] Loaded database with identities: ")
    for k,v in dff.faces_in_database().items():
        print(k if v==1 else f"{k} x{v}")

    # Initialize the USB webcam (0 is usually the default camera index)
    filename = "videos/Movie on 22-02-2026 at 19.47.mov"
    cap = cv2.VideoCapture(filename)

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
        now = time.time()
        dfaces = process_frame(frame)
        display_frame = draw_results(frame, dfaces)
        # Calculate and update FPS
        current_fps = calculate_fps()
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        
        # Display everything over the video feed.
        cv2.imshow('Face Rec Running', display_frame)
        
        # sound logic
        new_knowns = [k for k,v in dff.faces.items() if v["frame_count"] == MIN_FRAME_COUNT and not k.startswith("Unknown")]
        for name in new_knowns:
            print(f"[LOGIC] New known detected: {name}")
            playsound(name)
            dff.faces[name]["frame_count"] += 1 # dont get stuck

        stable_faces = [(k, v["gender"]) for k,v in dff.faces.items() if v["frame_count"] >= MIN_FRAME_COUNT]
        if len(stable_faces) > prev_stable_count:
            print(f"[LOGIC] Stable faces count changed {len(stable_faces)} (Previous: {prev_stable_count})")
            if len(stable_faces) == 1 :
                if stable_faces[0][1]=="Man":
                    playsound("unknown_m")
                else:
                    playsound("unknown_f")
            elif len(stable_faces) == 2:
                playsound("unknown_duo")
            elif len(stable_faces) > 2:
                playsound("unknown_group")
    
        prev_stable_count = len(stable_faces)
            
        # Break the loop and stop the script if 'q' is pressed
        if cv2.waitKey(1) == ord("q"):
            break
        if cv2.waitKey(1) == ord("s"):
            print("Saving capture.jpg")
            filename = os.path.join("tmp", f"capture_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.jpg")
            cv2.imwrite(filename, frame)
            q.put("camera-shutter-199580.mp3")


    # By breaking the loop we run this code here which closes everything
    q.put("DONE")  # Signal the sound loop to exit
    
    cap.release()
    cv2.destroyAllWindows()
