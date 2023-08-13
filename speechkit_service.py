from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        api_key='AQVNzOhwZYDd7ASl3jlbcCUNlmEmtdMwFPXuLcal'
    )
)

def recognize(audio):
    model = model_repository.recognition_model()

    # Задайте настройки распознавания.
    model.model = 'general'
    model.language = 'ru-RU'
    model.audio_processing_type = AudioProcessingType.Full

    # Распознавание речи в указанном аудиофайле и вывод результатов в консоль.
    result = model.transcribe_file(audio)
    text_res = ""
    for c, res in enumerate(result):
        text_res += res.raw_text + "\n"
    return text_res
