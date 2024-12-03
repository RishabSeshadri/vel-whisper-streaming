import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import whisper
import time
import httpx
from ollama_python.endpoints import GenerateAPI
from transformers import LlamaTokenizer


# Define the API endpoint and model
API_URL = "http://localhost:8000/generate"
MODEL_NAME = "llama-3.2:1b"

# Define the input prompt
prompt = ""

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Tokenize the input prompt
tokenized_prompt = tokenizer(prompt, return_tensors="pt")

audio_buffer = np.array([],dtype=np.float32)
model = whisper.load_model("medium.en") 

def process_audio():
    global audio_buffer
    while True:
        if len(audio_buffer) >= 2000:
            audio_data = audio_buffer.copy()
            transcript = model.transcribe(audio=audio_data,condition_on_previous_text=False,word_timestamps=False)
            if not transcript['text']:
                continue

            payload = {
                "model": MODEL_NAME,
                "prompt": transcript['text'],
                "options": {
                    "max_tokens": 50,
                    "temperature": 0.7
                },
                "pad_token_id": tokenizer.pad_token_id,
                "attention_mask": tokenized_prompt['attention_mask'].tolist()
            }

            try:
                response = httpx.post(API_URL, json=payload, timeout=60.0)
                response.raise_for_status()

                # Parse and print the response
                result = response.json()
                print("Llama 3.2 Response:", result.get("generated_text", "No response text found"))
            except httpx.TimeoutException:
                print("The request timed out. The model might be taking too long to respond.")
            except httpx.HTTPError as exc:
                print(f"HTTP Error occurred: {exc}")
            except Exception as e:
                print(f"An error occurred: {e}")
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

root = tk.Tk()
root.title("Microphone Recorder")
root.geometry("300x150")
start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.pack(pady=10)
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_button.pack(pady=10)
root.mainloop()

"""
def chat(content):
    api = GenerateAPI(base_url="http://localhost:8000", model="mistral")

    try:
        # Sending prompt to the API
        response = api.generate(
            prompt=content,
            images=None,
            options=None,
            system=None,
            stream=False,
            format="json",
            template=None,
            context=None,
            raw=False
        )

        # Handling the API response
        for res in response:
            if isinstance(res, dict) and 'generated_text' in res:
                print("Llama API Response:", res['generated_text'])  # <-- Process the response properly
            else:
                print("Unexpected response format:", res)
    except httpx.HTTPError as e:
        print(f"HTTP Error occurred: {e}")  # <-- Handling HTTP errors
    except Exception as e:
        print(f"An error occurred: {e}")  # <-- Handling any other errors


def launch_thread(func, *args):
    thread = threading.Thread(target=func, args=args, daemon=True)
    thread.start()
"""
"""
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,v
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])

model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token="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")

pipeline = VoiceActivityDetection(segmentation=model)


HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}

pipeline.instantiate(HYPER_PARAMETERS)

"""