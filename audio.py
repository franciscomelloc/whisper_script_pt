import os
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from datasets import load_dataset


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# class Audio:

#     def __init__(self):
#         self.recognizer = sr.Recognizer()

#     def import_audio(self, folder_location):
#         audio_files = []
#         for file in os.listdir(folder_location):
#             if file.endswith(".wav"):
#                 audio_files.append(os.path.join(folder_location, file))
#         return audio_files

#     def whisper_t2s(self, audio_file):
#         pass

class Audio:

    def __init__(self):
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to("cuda" if torch.cuda.is_available() else "cpu")

    def whisper_t2s(self, audio_path):
        # load audio file
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        input_features = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features

        # generate token ids
        predicted_ids = self.model.generate(input_features)
        
        # decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription