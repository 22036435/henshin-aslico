from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
from threading import Thread, Event
import openai
import random
import time
import subprocess
import csv
import pandas as pd
import speech_recognition as sr
#import pyttsx3

'''
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs available: ", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available.")
'''

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main page with a button to start the experience."""
    return render_template('index.html')

# Initialize Mediapipe for pose detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = tf.keras.models.load_model('action.h5')

actions = np.array(['better_US', 'better_UK', 'better_TR'])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Customize this function based on how you want to visualize the landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Prediction Visualizer
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

sentence_generation_event = Event()

current_sentence = "Generated sentences will appear here..."
generate_thread = None

# Global variable to keep track of the speech process
speech_process = None

def speak_macos(text):
    global speech_process
    # Ensure any existing speech process is stopped before starting a new one
    stop_speaking()
    # Use Popen to execute the say command in a non-blocking manner
    speech_process = subprocess.Popen(['say', text])

def stop_speaking():
    global speech_process
    # Check if there is an ongoing speech process
    if speech_process is not None:
        # Terminate the process
        speech_process.terminate()
        # Wait for process termination to ensure cleanup
        speech_process.wait()
        # Reset the speech process variable
        speech_process = None

'''# Global pyttsx3 engine initialization
tts_engine = pyttsx3.init()

def speak(text):
    global tts_engine
    tts_engine.say(text)
    tts_engine.runAndWait()


def stop_speaking():
    tts_engine.stop()'''

openai.api_key = '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'

def load_dataset_from_excel(file_path):
    """Load sentences from an Excel file into a list."""
    sentences = []
    df = pd.read_excel(file_path, usecols=[0], engine='openpyxl')  # Assuming the sentence is in the first column
    sentences = df[df.columns[0]].tolist()
    return sentences

def async_generate_similar_sentences(dataset, output_var, stop_event):
    global current_sentence
    while not stop_event.is_set():
        sentence_data = random.choice(dataset)  # Randomly select a sentence from the dataset
        current_sentence = sentence_data  # Update the GUI with the selected sentence
        speak_macos(sentence_data)  # Speak out the selected sentence
        time.sleep(2)  # Pause for 2 seconds before repeating, adjust the sleep time as needed

def convert_to_mindful(sentence):
    """Convert a sentence to a more mindful tone using an updated approach."""
    prompt = (
        "Please convert the following sentence into a more mindful and neutral tone, focusing on positive or neutral wording.\n\n"
        "Original: \"{}\"\n"
        "Mindful Version:".format(sentence)
    )
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Update to the latest version if available
            prompt=prompt,
            max_tokens=60,
            n=1,
            stop="\n",
            temperature=0.3,  # Lower temperature for less randomness
            presence_penalty=0.5,  # Encourage new concepts and discourage repetition
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred while converting to neutral tone: {e}")
        return sentence

def start_generating(dataset, stop_event):
    global generate_thread
    stop_event.clear()
    generate_thread = Thread(target=async_generate_similar_sentences, args=(dataset, current_sentence, stop_event))
    generate_thread.start()

def stop_generating(stop_event):
    global generate_thread, current_sentence
    stop_speaking()
    stop_event.set()

    if generate_thread is not None:
        generate_thread.join()
        generate_thread = None

    if current_sentence:
        mindful_sentence = convert_to_mindful(last_sentence)
        current_sentence = mindful_sentence
        speak_macos(mindful_sentence)

def gen_frames():

    global current_sentence

    cap = cv2.VideoCapture(0)
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Consider the last 30 frames

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                    # Additional check for "better_UK" detection
                    if "better_UK" in sentence:
                        # Detected "better_UK", now trigger stop_generating if it's running
                        if sentence_generation_event.is_set():
                            print('what is happeninnggggg')
                            stop_speaking()
                            stop_generating(sentence_generation_event)
                        else:
                            print('not on set')
                            pass

                image = prob_viz(res, actions, image, colors)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # Convert the processed image to JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Global variable to manage the background listening thread
listening_thread = None

def listening_task(stop_listening_command="stop listening", switch_command="switch"):

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        r.adjust_for_ambient_noise(source)
        print("Set up complete. Start speaking.")

        try:
            while True:
                print("Listening...")
                audio = r.listen(source, timeout=5)
                try:
                    text = r.recognize_google(audio)
                    print(f"You said: {text}")
                    if switch_command in text.lower():
                        print("Switch command detected.")
                        if sentence_generation_event.is_set():
                            stop_speaking()
                            stop_generating(sentence_generation_event)
                        else:
                            print('not on set')
                            pass
                    elif stop_listening_command in text.lower():
                        print("Stop command detected. Exiting...")
                        break
                except sr.UnknownValueError:
                    print("Google Web Speech could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Web Speech service; {e}")
        except KeyboardInterrupt:
            print("Terminating transcription.")

@app.route('/start_listening')
def start_listening_route():
    global listening_thread
    if listening_thread is None or not listening_thread.is_alive():
        # Adjust the commands as needed
        listening_thread = Thread(target=listening_task, args=("stop listening", "switch"))
        listening_thread.start()
        return jsonify({"message": "Listening started"}), 200
    else:
        return jsonify({"message": "Listening is already running"}), 200

@app.route('/start_generating')
def start_generating_route():
    # Placeholder values for dataset and output_var
    dataset = load_dataset_from_excel('FeedTheSpeech.xlsx')
    start_generating(dataset, sentence_generation_event)
    return jsonify({"message": "Generation started"})

@app.route('/stop_speaking')
def stop_speaking_route():
    stop_speaking()
    return jsonify({"message": "Speech stopped"})

@app.route('/get_current_sentence')
def get_current_sentence():
    global current_sentence
    return jsonify({"current_sentence": current_sentence})

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)