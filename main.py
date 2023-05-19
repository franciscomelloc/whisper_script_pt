import concurrent.futures
from audio import *

# def main():


def main():

    audio = Audio()
    
    for audio_file in os.listdir('audio'):
        if audio_file.endswith('.wav'):  # assuming the audio files are .wav
            transcription = audio.whisper_t2s(os.path.join('audio', audio_file))
            print(f'Transcription for {audio_file}: {transcription}')

            # write transcription to a .txt file
            with open(f'texts/{os.path.splitext(audio_file)[0]}.txt', 'w') as text_file:
                text_file.write(transcription[0])

   
    # audio = Audio()

    # audio_files = audio.import_audio('audio')

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     texts = list(executor.map(audio.whisper_t2s, audio_files))
    #      #store as .txt file

    # for i, audio_file in enumerate(audio_files):
    #     text = audio.whisper_t2s(audio_file)
    #     with open(f"texts/text_{i}.txt", "w", encoding='utf-8') as f:
    #         f.write(text[0])


if __name__ == '__main__':
    main()