from whisper_online import *

src_lan = "en"  # source language
tgt_lan = "en"  # target language  -- same as source for ASR, "en" if translate task is used

asr = FasterWhisperASR(lan, "large-v2")  # loads and wraps Whisper model
# set options:
# asr.set_translate_task()  # it will translate from lan into English
# asr.use_vad()  # set using VAD

online = OnlineASRProcessor(asr)  # create processing object with default buffer trimming option

while audio_has_not_ended:   # processing loop:
	a = # receive new audio chunk (and e.g. wait for min_chunk_size seconds first, ...)
	online.insert_audio_chunk(a)
	o = online.process_iter()
	print(o) # do something with current partial output
# at the end of this audio processing
o = online.finish()
print(o)  # do something with the last output


online.init()  # refresh if you're going to re-use the object for the next audio