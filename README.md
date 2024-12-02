# VEL Whisper Streaming
----------------------------------------------
The main purpose of this research project is to test and work to reduce latency in speech with chat-based AI models to be later implemented into VR NPCs.

### How to Run:
----------------------------------------------
Ensure all the libraries are correctly installed (i.e. openai-whisper, transformers, etc.), and Python 3 is being used. You can simply run the entire pipeline by running:
```$ python3 streamer_with_break.py```

### Speech-to-text Experiments
----------------------------------------------
The experiments were generally split into two parts - first, improving the efficiency of the speech-to-text algorithm from Whisper, and second, to improve the responses fed to Llama in a way that could allow a streaming or near-streaming simulation.

Significant improvements were found with the first part - the first approach used was to pick the largest and most efficient model, 'Turbo'. While this worked extremely well on the RTX 4090 GPUs, that was not the most accessible option. However, Whisper's 'Tiny' model was not accurate enough to be used for a direct stream into Llama. Additionally, when a user was not actually talking, the tiny model (as compared to the other versions) was much more susceptible for reading noise and creating random words. 

The next step was to process the input multiple times to allow for a better read on what a user was saying, as used in this ASR [https://www.youtube.com/watch?v=_spinzpEeFM] application.

When a word/sentence was processed 3 times repeatedly in the audio buffer, it was completely moved into a 'completed' buffer. This allowed for a working transcript of a user to be repeatedly updated, but only with confirmed words. Even with the use of the less-accurate 'Tiny' model, the results were extremely accurate when working with a mid-range microphone.

Finding out where to split the sentence was an important note - if the sentence was split in the middle, the context may be lost, which means that Whisper may not properly interpret the following words either. In order to prevent this, splits were performed at the end of a sentence - a period, comma, or exclamation mark was a marker of the end of the said sentence.

### Llama Experiments 
----------------------------------------------
The second half of the experimentation was with Llama, and finding out how to properly stream the messages into the model. The majority of the issues that were had with this section had to do with how a 'streaming' mode could be achieved with Llama. The sentences were fed as they came in from the whisper stream, however this led to a massive downtime while a sentence was being processed and responded to by Llama.

Initially, any sentences that were said while content was being processed by Llama would be placed into a buffer and sent on the next request - however, since the main purpose/application of this project was to work with Non-Playable Characters, this was abandoned in favor of a more conversational buffer that only read when Llama was not actively processing data. Additionally, this meant handling the empty-speech segments that resulted in Whisper sending periods.  

### Future steps
----------------------------------------------
There are plenty of possible next steps:
  - **Exploring the integration into Unity:** Unity integration may require optimizing server-client communication for real-time interactions and patching the possible latency issues in VR
  - **Exploring other models:** There may be other models besides Llama that would have better results - there were issues with the way that Llama responded to prompts, answering in non-specific or unrelated answers. This seemed to be an issue with the model itself or how the request is being made, rather than the words being sent to it.
  - **Figuring out streaming:** This current model does not actually use a proper streaming method, but figuring out how to feed data to Llama constantly and then mark when the request is ready to be made would be a good next step to follow.
----------------------------------------------
*Authors: Rishab Seshadri*