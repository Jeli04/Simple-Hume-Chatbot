from dotenv import dotenv_values
import streaming_test
import interpret_test
import asyncio
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from hume import HumeStreamClient
from hume.models.config import FaceConfig
import time
import cv2

env_vars = dotenv_values('.env')

# move this to a env variable
llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'], temperature=0.5)
memory = ConversationBufferMemory(return_messages=True, ai_prefix="AI friend")

template = """Your name is Echo and the humans name is Jerry. The following is a conversation between a human and an AI who have been with each other since birth. The AI is able to detect the humans current facial emotion. Both have known each for a logn time and have developed a very close relationship. The language used by the AI isnt quite robotic, but also incoorpates a mix of human style talk and robotic talk together. When taking account an emotion treat it like how another human would if they saw that persons facial expression. Be firm and dont back down with your opinions as an AI. Make sure to have a funny personality that is able to shine through and make you stand out.

Current converstation:
{history}
Conversation:
Human Emotion: {emotion}
Human: {input}
AI friend:"""
prompt = PromptTemplate(input_variables=["history", "input", "emotion"], template=template)


async def main():
    lock = asyncio.Lock()
    print("starting conversation")
    conversation = ConversationChain(llm=llm, verbose=True, memory=interpret_test.ExtendedConversationBufferMemory(extra_variables=["emotion"]), prompt=prompt)
    start_time = time.time()
    client = HumeStreamClient(env_vars['HUMEAI_API'])
    config = FaceConfig(identify_faces=True)
    cap = cv2.VideoCapture(0)
    emotion = ""

    input_message = await asyncio.get_event_loop().run_in_executor(None, input, 'Enter message: ')
    async with client.connect([config]) as socket:
        while(input_message != "exit"):
            emotion = await asyncio.create_task(streaming_test.emotion_stream(lock=lock, socket=socket, cap=cap, current_time=time.time(), start_time=start_time, elapsed_time=time.time()-start_time))

            if(emotion != ""):
                async with lock:
                    result = conversation({"input": input_message, "emotion": emotion})
            else:
                result = conversation({"input": input_message, "emotion": "neutral"})
            print(result["response"])
            input_message = await asyncio.get_event_loop().run_in_executor(None, input, 'Enter message: ')

    cap.release()
    cv2.destroyAllWindows()

asyncio.run(main())