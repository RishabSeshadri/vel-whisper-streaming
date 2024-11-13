import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import whisper
import time
audio_buffer = np.array([],dtype=np.float32)
model = whisper.load_model("tiny.en") 

full_transcript = np.array([],dtype=np.float32)

def process_audio():
    global audio_buffer
    while True:
        if len(audio_buffer) >= 2000:
            audio_data = audio_buffer.copy()
            transcript = model.transcribe(audio=audio_data,condition_on_previous_text=False,word_timestamps=False)
            if transcript['text']:
                print(transcript['text'], flush=True)
                #audio_buffer = np.array([],dtype=np.float32)
        time.sleep(.1)

# Start the processing thread
processing_thread = threading.Thread(target=process_audio, daemon=True)
processing_thread.start()

def callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = np.append(audio_buffer,np.squeeze(indata))

def start_recording():
    global stream
    global start_button
    global stop_button
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

    stream = sd.InputStream(samplerate=16000, device=1,channels=1, callback=callback, dtype=np.float32)
    stream.start()

def stop_recording():
    global audio_buffer
    global stream
    global start_button
    global stop_button
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    audio_buffer = np.array([],dtype=np.float32)
    stream.stop()
    stream.close()
    print(full_transcript)

root = tk.Tk()
root.title("Microphone Recorder")
root.geometry("300x150")
start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.pack(pady=10)
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_button.pack(pady=10)
root.mainloop()


"""
# Server settings
HOST = 'localhost'
PORT = 3000

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}...")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)  # Adjust buffer size as needed
                if not data:
                    break
                # Here you can process the received audio data (e.g., save to file, stream to Whisper, etc.)
                print(f"Received {len(data)} bytes of audio data")

if __name__ == "__main__":
    start_server()
"""