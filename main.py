import os
import soundfile as sf
import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_model_and_processor():
    
    """
    Carrega o modelo e o processador do Whisper ASR (Automatic Speech Recognition).

    Esta função primeiro verifica se a GPU está disponível para uso e configura o dispositivo
    apropriado. Em seguida, carrega o processador e o modelo do Whisper ASR da OpenAI.

    Retorna:
        processor (WhisperProcessor): O processador do Whisper ASR, utilizado para 
                                      processar dados de áudio para o modelo.
        model (WhisperForConditionalGeneration): O modelo de reconhecimento automático de voz Whisper.
        device (str): O dispositivo que será usado para rodar o modelo ("cuda:0" para GPU, "cpu" para CPU).
    """

    # verifica o dispositivo
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # carrega o modelo e o processador
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
    
    return processor, model, device

def get_forced_decoder_ids(processor):

    """
    Obtém os IDs de decodificação forçados para o idioma português.

    Essa função utiliza o processador do Whisper ASR para obter os IDs de decodificação 
    forçados, que são usados para forçar o modelo a reconhecer o idioma português.

    Args:
        processor (WhisperProcessor): O processador do Whisper ASR, utilizado para processar 
                                      dados de áudio para o modelo.

    Retorna:
        forced_decoder_ids (torch.Tensor): Tensor com os IDs de decodificação forçados 
                                           para o idioma português.
    """
    
    # define os IDs de decodificação forçados para o português
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="portuguese", task="transcribe")
    
    return forced_decoder_ids

def get_wav_files(directory_path):

    """
    Obtém uma lista de arquivos .wav em um diretório especificado.

    Esta função percorre todos os arquivos em um diretório dado e retorna 
    uma lista contendo apenas aqueles que terminam com a extensão .wav.

    Args:
        directory_path (str): O caminho do diretório a ser examinado.

    Retorna:
        wav_files (list of str): Uma lista de strings contendo os nomes dos arquivos .wav.
    """

    # obtém a lista de arquivos .wav no diretório
    wav_files = [f for f in os.listdir(directory_path) if f.endswith(".wav")]
    
    return wav_files

def create_directory(directory_name):

    """
    Cria um diretório se ele ainda não existir.

    Esta função usa a função os.makedirs para criar um novo diretório. 
    Se o diretório já existir, a função não fará nada graças ao argumento exist_ok=True.

    Args:
        directory_name (str): O nome (ou caminho) do diretório a ser criado.
    """

    # cria um diretório se ele não existir
    os.makedirs(directory_name, exist_ok=True)

def resample_audio(audio_array, sampling_rate):

    """
    Converte o áudio para mono e o redimensiona para 16000 Hz se necessário.

    Esta função primeiro verifica se o áudio está em estéreo (duas dimensões) e, 
    em caso afirmativo, converte para mono fazendo a média das duas trilhas. 
    Em seguida, verifica a taxa de amostragem e, se não for 16000 Hz, 
    redimensiona o áudio para essa taxa de amostragem.

    Args:
        audio_array (numpy.ndarray): O array de áudio a ser processado.
        sampling_rate (int): A taxa de amostragem do áudio.

    Retorna:
        audio_array (numpy.ndarray): O array de áudio processado.
        sampling_rate (int): A nova taxa de amostragem do áudio.
    """

    # se estéreo, converte para mono
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # redimensiona para 16000 Hz se necessário
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    
    return audio_array, sampling_rate

def transcribe(audio_array, sampling_rate, processor, model, device, forced_decoder_ids):
    
    """
    Converte um array de áudio em transcrições de texto.

    Esta função divide o array de áudio em partes de 30 segundos, convertendo cada parte em 
    recursos de entrada para o modelo Whisper ASR, gerando IDs de token previstos e, em seguida,
    decodificando esses IDs em texto. As transcrições de cada parte são retornadas em uma lista.

    Args:
        audio_array (numpy.ndarray): O array de áudio a ser transcrito.
        sampling_rate (int): A taxa de amostragem do áudio.
        processor (WhisperProcessor): O processador do Whisper ASR.
        model (WhisperForConditionalGeneration): O modelo Whisper ASR.
        device (str): O dispositivo para o qual os tensores devem ser enviados ("cuda:0" ou "cpu").
        forced_decoder_ids (torch.Tensor): Os IDs de decodificação forçados para o idioma português.

    Retorna:
        transcriptions (list of str): A lista de transcrições de texto para cada parte do áudio.
    """

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
        if transcription[0].strip():  # check if the transcription is not empty
            transcriptions.append(transcription[0])  # append the transcription of the current chunk to the list
        print(transcriptions)
    return ' '.join(transcriptions)

def write_transcription_to_file(wav_file, transcription):
    
    """
    Escreve a transcrição em um arquivo .txt no diretório 'texts'.

    Esta função cria um nome de arquivo .txt correspondente ao arquivo .wav fornecido e escreve 
    a transcrição fornecida neste arquivo .txt. O arquivo .txt é armazenado no diretório 'texts'. 
    Se a transcrição for bem-sucedida, uma mensagem será impressa indicando isso.

    Args:
        wav_file (str): O nome do arquivo .wav que foi transcrito.
        transcription (list of str): A transcrição a ser escrita no arquivo .txt. 
                                     É uma lista com um único elemento de string.

    """

    # escreve a transcrição em um arquivo .txt no diretório 'texts'
    txt_file = os.path.splitext(wav_file)[0] + '.txt'  # altera a extensão .wav para .txt
    with open(os.path.join("texts", txt_file), "w") as f:
        f.write(transcription)  # a transcrição é uma lista com um único elemento de string

    print(f"Transcription for {wav_file} written to {txt_file}")

def main():

    """
    Função principal que orquestra o fluxo de transcrição de áudio para texto.

    Esta função carrega o modelo e o processador do Whisper ASR, obtém os IDs de decodificação 
    forçados para o português, lista todos os arquivos .wav no diretório fornecido, cria um 
    diretório 'texts' se ainda não existir e processa cada arquivo de áudio, transcrevendo o 
    áudio para texto e escrevendo a transcrição em um arquivo .txt correspondente no diretório 'texts'.

    """

    directory_path = "audio"
    processor, model, device = load_model_and_processor()
    forced_decoder_ids = get_forced_decoder_ids(processor)
    wav_files = get_wav_files(directory_path)
    create_directory("texts")
    
    # processa amostras de áudio
    for wav_file in wav_files:
        # lê o arquivo de áudio
        audio_array, sampling_rate = sf.read(os.path.join(directory_path, wav_file))
        audio_array, sampling_rate = resample_audio(audio_array, sampling_rate)
        all_transcription = transcribe(audio_array, sampling_rate, processor, model, device, forced_decoder_ids)
        write_transcription_to_file(wav_file, all_transcription)

if __name__ == "__main__":
    main()
