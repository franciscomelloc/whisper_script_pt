import os
import soundfile as sf
import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_model_and_processor():
    # check device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
    
    return processor, model, device

def get_forced_decoder_ids(processor):
    # set the forced decoder ids to Portuguese
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="portuguese", task="transcribe")
    
    return forced_decoder_ids

def get_wav_files(directory_path):
    # get list of .wav files in directory
    wav_files = [f for f in os.listdir(directory_path) if f.endswith(".wav")]
    
    return wav_files

def create_directory(directory_name):
    # create a directory if it doesn't exist
    os.makedirs(directory_name, exist_ok=True)

def resample_audio(audio_array, sampling_rate):
    # if stereo, convert to mono
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # resample to 16000 Hz if necessary
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    
    return audio_array, sampling_rate

def transcribe(audio_array, sampling_rate, processor, model, device, forced_decoder_ids):
    # convert audio array into chunks of 30 seconds
    chunk_size = 30 * sampling_rate  # chunk size in samples
    chunks = [audio_array[i:i + chunk_size] for i in range(0, len(audio_array), chunk_size)]

    transcriptions = []
    for chunk in chunks:
        # convert audio sample to features
        input_features = processor(chunk, 
                                   sampling_rate=sampling_rate, 
                                   return_tensors="pt").input_features.to(device)

        # generate token ids
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        # decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcriptions.append(transcription[0])  # append the transcription of the current chunk to the list
        
    return transcriptions
    #return " ".join(transcriptions)  # join all transcriptions into a single string


def write_transcription_to_file(wav_file, transcription):
    # write transcription to a .txt file in the 'texts' directory
    txt_file = os.path.splitext(wav_file)[0] + '.txt'  # change .wav extension to .txt
    with open(os.path.join("texts", txt_file), "w") as f:
        f.write(transcription[0])  # transcription is a list with a single string element

    print(f"Transcription for {wav_file} written to {txt_file}")

def main():
    directory_path = "audio"
    processor, model, device = load_model_and_processor()
    forced_decoder_ids = get_forced_decoder_ids(processor)
    wav_files = get_wav_files(directory_path)
    create_directory("texts")
    
    # process audio samples
    for wav_file in wav_files:
        # read audio file
        audio_array, sampling_rate = sf.read(os.path.join(directory_path, wav_file))
        audio_array, sampling_rate = resample_audio(audio_array, sampling_rate)
        transcription = transcribe(audio_array, sampling_rate, processor, model, device, forced_decoder_ids)
        write_transcription_to_file(wav_file, transcription)

if __name__ == "__main__":
    main()
