from dotenv import dotenv_values
from tests import streaming_test as streaming
from tests import interpret_test as interpret
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from hume import HumeStreamClient
from hume.models.config import FaceConfig
from pynput.keyboard import Key, Listener
from pvrecorder import PvRecorder
from gtts import gTTS
from playsound import playsound
import asyncio
import threading
import wave
import struct
import time
import cv2
import os
import base64
from interpret import interpret

env_vars = dotenv_values('.env')

llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'], temperature=0.5)
memory = ConversationBufferMemory(return_messages=True, ai_prefix="AI friend")

template = """Your name is Echo and the human you are paired with is named Jerry. The following is a conversation between a human and an AI who have been with each other since birth. The AI is able to detect the humans current facial emotion. Both have known each for a logn time and have developed a very close relationship. The language used by the AI isnt quite robotic, but also incoorpates a mix of human style talk and robotic talk together. When taking account an emotion treat it like how another human would if they saw that persons facial expression. Be firm and dont back down with your opinions as an AI. Make sure to have a funny personality that is able to shine through and make you stand out.

Current converstation:
{history}
Conversation:
Human Emotion: {emotion}
Human: {input}
AI friend:"""
prompt = PromptTemplate(input_variables=["history", "input", "emotion"], template=template)

recorder = PvRecorder(device_index=1, frame_length=512)
recording = False
audio = []

# For the hume stream
lock = asyncio.Lock()
start_time = time.time()
client = HumeStreamClient(env_vars['HUMEAI_API'])
config = FaceConfig(identify_faces=True)
cap = cv2.VideoCapture(0)
emotion = ""


def record_voice():
    global recorder, recording, audio
    while recording:
        frame = recorder.read()
        audio.extend(frame)

    recorder.stop() # stop recording
    print("Ending recording....")

    with wave.open("output.wav", 'w') as f:
        f.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        f.writeframes(struct.pack("h" * len(audio), *audio))

    audio = [] # reset audio

    # call the interpret function 
    result = interpret("output.wav", emotion=emotion)
    os.remove("output.wav")
    
    # play the result 
    result_voice = gTTS(text=result["response"], lang='en', slow=False)
    result_voice.save("result.mp3")
    # playsound(os.path.dirname(__file__) + '\\result.mp3')
    print(result["response"])
    playsound("result.mp3")
    os.remove("result.mp3")

# modify the code so the steam stops when space bar is hit
async def stream_emotion():
    global emotion, lock, start_time, client, config, cap, recording

    start_time = time.time()
    emotion = ""
    async with client.connect([config]) as socket:
        while recording:     # modifiy this line too allow continous streaming
            # record emotion
            current_time = time.time()
            elapsed_time = current_time - start_time

            # print("detecting emotion")
            ret, frame = cap.read()
            _, frame_bytes = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(frame_bytes.tobytes())
            result = await socket.send_bytes(frame_base64)
                    
            # interval is the time
            if(elapsed_time >= 2):
                if "warning" in result["face"]:
                    if emotion == "":
                        emotion = "neutral"    
                    # print("No face detected")
                else:
                    async with lock:
                        emotion = streaming.get_likely_emotion(result["face"]["predictions"][0]["emotions"])
                        # print(emotion)
        
                start_time = current_time

 
def run_stream_emotion_in_thread():
    asyncio.run(stream_emotion())

def on_key_press(key):
    global recorder, recording

    # temp code  
    if key == Key.esc:
        quit()

    if key == Key.space:
        if recording:
            print("Calling interpret....")
            recording = False
        else:
            print("Recording....")
            recording = True
            recorder.start()  
            threading.Thread(target=run_stream_emotion_in_thread).start()
            threading.Thread(target=record_voice).start()

async def main():
    lock = asyncio.Lock()
    print("starting conversation")
    conversation = ConversationChain(llm=llm, verbose=True, memory=interpret.ExtendedConversationBufferMemory(extra_variables=["emotion"]), prompt=prompt)
    start_time = time.time()
    client = HumeStreamClient(env_vars['HUMEAI_API'])
    config = FaceConfig(identify_faces=True)
    cap = cv2.VideoCapture(0)
    emotion = ""

    input_message = await asyncio.get_event_loop().run_in_executor(None, input, 'Enter message: ')
    async with client.connect([config]) as socket:
        while(input_message != "exit"):
            emotion = await asyncio.create_task(streaming.emotion_stream(lock=lock, socket=socket, cap=cap, current_time=time.time(), start_time=start_time, elapsed_time=time.time()-start_time))

            if(emotion != ""):
                async with lock:
                    result = conversation({"input": input_message, "emotion": emotion})
            else:
                result = conversation({"input": input_message, "emotion": "neutral"})
            print(result["response"])
            input_message = await asyncio.get_event_loop().run_in_executor(None, input, 'Enter message: ')

    cap.release()
    cv2.destroyAllWindows()

# asyncio.run(main())


with Listener(on_press = on_key_press) as listener:  
    listener.join()


"""
New project structure

intialize all global variables

(in a different py file)
interpret function
    uses whisper to convert the temp file to text
    delete the temp file
    send the text to GPT to get the response
    returns a string with the get_message function

get_message function
    
stream_emotion function

on_press function
    if keypress is +:
        if already recording 
            end voice recording and stream to hume
            store the recording as a temp file
            await send to interpret function with emotion
        else
            start voice recording and stream to hume

listen for keypresses
    if keypress call the on_press function

"""