import asyncio
import cv2
import base64
import time
import string
from dotenv import dotenv_values
from hume import HumeStreamClient, StreamSocket
from hume.models.config import FaceConfig

env_vars = dotenv_values('.env')

async def emotion_stream(lock=None):
    interval = 5
    start_time = time.time()
    client = HumeStreamClient(env_vars['HUMEAI_API'])
    config = FaceConfig(identify_faces=True)
    cap = cv2.VideoCapture(0)
    async with client.connect([config]) as socket:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            ret, frame = cap.read()
            _, frame_bytes = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(frame_bytes.tobytes())
            result = await socket.send_bytes(frame_base64)
            
            # Displays the webcam
            cv2.imshow('Webcam', frame)

            if(elapsed_time >= interval):
                if "warning" in result["face"]:
                    print("No face detected")
                else:
                    emotion = get_likely_emotion(result["face"]["predictions"][0]["emotions"])
                    print(emotion)
                    
                start_time = current_time

            # Check for any key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(elapsed_time)

        # Release the video capture and close the socket
        cap.release()
        cv2.destroyAllWindows()


async def emotion_stream(lock : asyncio, socket : HumeStreamClient, cap:cv2.VideoCapture, current_time:float, start_time:float, elapsed_time:float): 
    print("detecting emotion")
    ret, frame = cap.read()
    _, frame_bytes = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(frame_bytes.tobytes())
    result = await socket.send_bytes(frame_base64)
            
    # interval is the time
    emotion = ""
    if(elapsed_time >= 5):
        if "warning" in result["face"]:
            print("No face detected")
        else:
            async with lock:
                emotion = get_likely_emotion(result["face"]["predictions"][0]["emotions"])
                print(emotion)
   
        start_time = current_time
    return emotion

def get_likely_emotion(emotions):
    highest_score = 0
    highest_emotion = ""
    for emotion in emotions:
        if emotion["score"] > highest_score:
            highest_score = emotion["score"]
            highest_emotion = emotion["name"]
    return highest_emotion

# asyncio.run(emotion_stream())

# Checks what webcams are available
# def check_webcams():
#     num_cameras = 0
#     while True:
#         cap = cv2.VideoCapture(num_cameras)
#         if not cap.isOpened():
#             break
#         _, _ = cap.read()  # Read a frame to check if the camera is working
#         cap.release()  # Release the camera
#         print(f"Webcam {num_cameras}: Available")
#         num_cameras += 1

#     if num_cameras == 0:
#         print("No webcams found.")

# check_webcams()