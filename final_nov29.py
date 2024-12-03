import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import whisper
import time
import httpx
from transformers import LlamaTokenizer

# Set up constants and initial variables
API_URL = "http://localhost:8000/generate"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
sample_rate = 16000
MAX_AUDIO_BUFFER_SIZE = sample_rate * 30  # Keep last 30 seconds of audio
audio_buffer = np.array([], dtype=np.float32)
full_transcript = ""  # Full confirmed transcript
unconfirmed_transcript = ""  # Unconfirmed, in-progress transcript
transcript_history = {}
confirmed_sentences = []
audio_offset = 0.0
last_confirmed_end_time = 0.0

# Initialize Whisper model
model = whisper.load_model("tiny.en")

# Initialize Llama tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Lock to control when audio processing should pause
processing_lock = threading.Lock()
is_waiting_for_response = False
is_recording_paused = False  # Flag to indicate whether recording should be paused

def process_audio():
    global audio_buffer, transcript_history, confirmed_sentences
    global full_transcript, unconfirmed_transcript, audio_offset, last_confirmed_end_time, is_waiting_for_response, is_recording_paused

    while True:
        if len(audio_buffer) >= sample_rate * 1:
            if is_waiting_for_response:
                # If waiting for Llama's response, pause audio processing
                time.sleep(0.1)
                continue

            if is_recording_paused:
                # If recording is paused, ignore audio input
                time.sleep(0.1)
                continue

            audio_data = audio_buffer.copy()
            transcript = model.transcribe(audio=audio_data, condition_on_previous_text=False, word_timestamps=False)
            
            if 'segments' in transcript and transcript['segments']:
                for segment in transcript['segments']:
                    text = segment['text'].strip()
                    start_time = segment['start'] + audio_offset
                    end_time = segment['end'] + audio_offset

                    # Detect a period as a potential pause or end-of-sentence
                    if text == ".":
                        # Treat this as a pause and handle it separately
                        send_waiting_message_to_llama()
                        continue  # Skip sending the period

                    if text.endswith('.') or text.endswith('?'):
                        # Confirm the sentence when it ends
                        if text in transcript_history:
                            transcript_history[text]['count'] += 1
                        else:
                            transcript_history[text] = {
                                'count': 1,
                                'start_time': start_time,
                                'end_time': end_time
                            }

                # Confirm sentences
                for text, data in list(transcript_history.items()):
                    if data['count'] >= 3 and text not in confirmed_sentences:
                        confirmed_sentences.append(text)
                        full_transcript += text + ' '
                        last_confirmed_end_time = max(last_confirmed_end_time, data['end_time'])

                        # Remove sentence from unconfirmed_transcript
                        unconfirmed_transcript = unconfirmed_transcript.replace(text, '').strip()

                        # Remove from transcript_history
                        del transcript_history[text]

                        # Print confirmed transcript
                        print('[CONFIRMED] ' + full_transcript.strip(), flush=True)

                        # Send confirmed sentence to Llama API
                        send_to_llama(text)

                # Remove confirmed audio from buffer
                if last_confirmed_end_time > audio_offset:
                    samples_to_remove = int((last_confirmed_end_time - audio_offset) * sample_rate)
                    audio_buffer = audio_buffer[samples_to_remove:]
                    audio_offset = last_confirmed_end_time

                # Update unconfirmed_transcript
                current_transcript = transcript['text'].strip()
                for sentence in confirmed_sentences:
                    current_transcript = current_transcript.replace(sentence, '')
                unconfirmed_transcript = current_transcript.strip()

                # Print unconfirmed transcript
                print(unconfirmed_transcript, flush=True)
        time.sleep(0.1)

def send_waiting_message_to_llama():
    """
    Sends a special message to the Llama API indicating the system is waiting for user input.
    """
    global is_waiting_for_response
    with processing_lock:
        is_waiting_for_response = True

    message = "Waiting for the user to speak..."
    print(f"Sending to Llama API: {message}")  # Debug message
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": message,
            "options": {
                "max_tokens": 250,
                "temperature": 0.2
            },
            "pad_token_id": tokenizer.pad_token_id
        }
        response = httpx.post(API_URL, json=payload, timeout=60.0)
        response.raise_for_status()

        # Parse and print the response
        result = response.json()
        print("Llama Response:", result.get("generated_text", "No response text found"))
    except httpx.TimeoutException:
        print("The request timed out. The model might be taking too long to respond.")
    except httpx.HTTPError as exc:
        print(f"HTTP Error occurred: {exc}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Unlock and allow audio processing to continue after the response is received
    with processing_lock:
        is_waiting_for_response = False

def send_to_llama(text):
    """
    Sends the confirmed text to Llama API and prints the response.
    """
    global is_waiting_for_response
    with processing_lock:
        is_waiting_for_response = True

    # Prepare the request payload for Llama API
    tokenized_prompt = tokenizer(text, return_tensors="pt")
    payload = {
        "model": MODEL_NAME,
        "prompt": text,
        "options": {
            "max_tokens": 50,
            "temperature": 0.7
        },
        "pad_token_id": tokenizer.pad_token_id,
        "attention_mask": tokenized_prompt['attention_mask'].tolist()
    }

    print(f"Sending to Llama API: {text}")  # Message to terminal
    try:
        response = httpx.post(API_URL, json=payload, timeout=60.0)
        response.raise_for_status()

        # Parse and print the response
        result = response.json()
        print("Llama Response:", result.get("generated_text", "No response text found"))

    except httpx.TimeoutException:
        print("The request timed out. The model might be taking too long to respond.")
    except httpx.HTTPError as exc:
        print(f"HTTP Error occurred: {exc}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Unlock and allow audio processing to continue after the response is received
    with processing_lock:
        is_waiting_for_response = False

def callback(indata, frames, time, status):
    """
    Callback function to store incoming audio samples.
    """
    global audio_buffer, audio_offset, is_waiting_for_response

    # If the system is waiting for a response, skip adding audio to the buffer
    if is_waiting_for_response:
        return

    audio_buffer = np.append(audio_buffer, np.squeeze(indata))

    # Limit audio buffer to last MAX_AUDIO_BUFFER_SIZE samples
    if len(audio_buffer) > MAX_AUDIO_BUFFER_SIZE:
        samples_to_remove = len(audio_buffer) - MAX_AUDIO_BUFFER_SIZE
        audio_buffer = audio_buffer[-MAX_AUDIO_BUFFER_SIZE:]
        audio_offset += samples_to_remove / sample_rate

def start_recording():
    """
    Starts audio recording.
    """
    global stream
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

    stream = sd.InputStream(samplerate=sample_rate, device=2, channels=1, callback=callback, dtype=np.float32)
    stream.start()

def stop_recording():
    """
    Stops audio recording and prints the final transcript.
    """
    global audio_buffer, stream
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    audio_buffer = np.array([], dtype=np.float32)
    stream.stop()
    stream.close()
    print(full_transcript)

# Initialize the Tkinter UI
root = tk.Tk()
root.title("Microphone Recorder")
root.geometry("300x150")

start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.pack(pady=10)
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_button.pack(pady=10)

# Start the audio processing in a separate thread
processing_thread = threading.Thread(target=process_audio, daemon=True)
processing_thread.start()
print(sd.query_devices())

root.mainloop()
