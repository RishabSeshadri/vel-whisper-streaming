import socket
import socket
from whisper_online import FasterWhisperASR, OnlineASRProcessor
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Server settings
HOST = 'localhost'
PORT = 3000

# Initialize Whisper
asr = FasterWhisperASR("en", "large-v2")
online = OnlineASRProcessor(asr)

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}...")

        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                
                # Pass the received audio data to Whisper
                online.insert_audio_chunk(data)
                result = online.process_iter()
                if result:
                    print(result)

            # Final output
            final_result = online.finish()
            print(final_result)

if __name__ == "__main__":
    start_server()



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