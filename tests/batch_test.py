from hume import HumeBatchClient
from hume.models.config import FaceConfig
from pprint import pprint
from dotenv import dotenv_values

env_vars = dotenv_values('.env')

client = HumeBatchClient(env_vars['HUMEAI_API'])

# starting a job
def predict_emotions(files):
    config = FaceConfig()
    job = client.submit_job(None, configs=[config], files=files)

    print(job)
    print("Running.....")
    job.await_complete()
    job.download_predictions("predictions.json")

    # checking the job status
    status = job.get_status()
    print(f"Job status: {status}")

    details = job.get_details()
    run_time_ms = details.get_run_time_ms()
    print(f"Job run time: {run_time_ms} miliseconds")

    # getting the results
    predictions = job.get_predictions()
    pprint(predictions)

    return predictions

def get_likely_emotion(all_predictions):
    emotions = []
    for predictions in all_predictions:
        prediction = predictions['results']['predictions'][0]['models']["face"]['grouped_predictions'][0]['predictions'][0]
        highest_score = 0
        highest_emotion = ""
        for emotion in prediction["emotions"]:
            if emotion["score"] > highest_score:
                highest_score = emotion["score"]
                highest_emotion = emotion["name"]
        emotions.append(highest_emotion)
    return emotions


# Specify the folder path
folder_path = 'pictures'
images = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Process the image file
        images.append(os.path.join(folder_path, filename))
        

for emotion in get_likely_emotion(predict_emotions(images)):
    print(emotion)