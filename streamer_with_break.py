import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import whisper
import time

audio_buffer = np.array([], dtype=np.float32)
model = whisper.load_model("tiny.en")

full_transcript = ""
transcribing_displayed = False  # *** Flag to check if 'Transcribing...' has been printed

transcript_history = {}
confirmed_sentences = []
sample_rate = 16000

audio_offset = 0.0  # Time in seconds of the start of audio_buffer
last_confirmed_end_time = 0.0

def process_audio():
    global audio_buffer
    global transcript_history
    global confirmed_sentences
    global sample_rate
    global full_transcript
    global audio_offset
    global last_confirmed_end_time
    global transcribing_displayed  # ***
    while True:
        if len(audio_buffer) >= sample_rate * 1:
            if not transcribing_displayed:  # ***
                print('Transcribing...', flush=True)  # ***
                transcribing_displayed = True  # ***

            audio_data = audio_buffer.copy()
            transcript = model.transcribe(audio=audio_data, condition_on_previous_text=False, word_timestamps=False)
            if 'segments' in transcript and transcript['segments']:
                # For each segment
                for segment in transcript['segments']:
                    text = segment['text'].strip()
                    start_time = segment['start'] + audio_offset
                    end_time = segment['end'] + audio_offset
                    if text.endswith('.') or text.endswith('?'):
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
                        # Update last_confirmed_end_time
                        last_confirmed_end_time = max(last_confirmed_end_time, data['end_time'])
                        # Remove from transcript_history
                        del transcript_history[text]
                        # Print confirmed transcript
                        print('[CONFIRMED] ' + full_transcript.strip(), flush=True)
                # Remove confirmed audio from buffer
                if last_confirmed_end_time > audio_offset:
                    samples_to_remove = int((last_confirmed_end_time - audio_offset) * sample_rate)
                    audio_buffer = audio_buffer[samples_to_remove:]
                    audio_offset = last_confirmed_end_time
        else:
            # Reset 'Transcribing...' flag when there is no audio to process
            if transcribing_displayed:  # ***
                transcribing_displayed = False  # ***
        time.sleep(.1)

# Start the processing thread
processing_thread = threading.Thread(target=process_audio, daemon=True)
processing_thread.start()

def callback(indata, frames, time, status):
    global audio_buffer
    global audio_offset
    audio_buffer = np.append(audio_buffer, np.squeeze(indata))
    # Limit audio_buffer to last MAX_AUDIO_BUFFER_SIZE samples
    MAX_AUDIO_BUFFER_SIZE = sample_rate * 30  # Keep last 30 seconds
    if len(audio_buffer) > MAX_AUDIO_BUFFER_SIZE:
        samples_to_remove = len(audio_buffer) - MAX_AUDIO_BUFFER_SIZE
        audio_buffer = audio_buffer[-MAX_AUDIO_BUFFER_SIZE:]
        audio_offset += samples_to_remove / sample_rate

def start_recording():
    global stream
    global start_button
    global stop_button
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

    stream = sd.InputStream(samplerate=16000, device=1, channels=1, callback=callback, dtype=np.float32)
    stream.start()

def stop_recording():
    global audio_buffer
    global stream
    global start_button
    global stop_button
    global transcribing_displayed  # ***
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    audio_buffer = np.array([], dtype=np.float32)
    stream.stop()
    stream.close()
    transcribing_displayed = False  # ***
    print('Final Transcript:', full_transcript.strip())

root = tk.Tk()
root.title("Microphone Recorder")
root.geometry("300x150")
start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.pack(pady=10)
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_button.pack(pady=10)
stop_button.config(state=tk.DISABLED)  # *** Disable stop button initially
root.mainloop()
    