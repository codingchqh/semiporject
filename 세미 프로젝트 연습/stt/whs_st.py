import sounddevice as sd
import numpy as np
import io
import wave
import whisper
import logging
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. WhisperSTT 클래스 정의
class WhisperSTT:
    def __init__(self, model_name="base", device=None):
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_path):
        # 오디오 파일 경로를 받아 텍스트 변환 결과 반환
        result = self.model.transcribe(audio_path)
        return result["text"]

def list_input_devices():
    devices = sd.query_devices()
    input_devices = []
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((idx, device['name']))
    return input_devices

def record_audio(duration_sec=5, fs=16000, device=None):
    """
    마이크에서 duration_sec초간 오디오 녹음 후 numpy int16 배열 반환
    device: None이면 기본 장치 사용, 아니면 장치 번호 지정
    """
    logger.info(f"Recording started: {duration_sec} seconds from device {device}")
    if device is None:
        recording = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16')
    else:
        recording = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait()
    logger.info("Recording finished")
    return recording.flatten()

def numpy_to_wav_bytes(audio_np, fs=16000):
    """
    numpy int16 배열을 WAV 형식 메모리 바이트 스트림으로 변환
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(fs)
        wf.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer

def transcribe_from_memory(model, wav_bytes_io):
    """
    메모리 WAV BytesIO 객체를 Whisper 모델에 입력해 텍스트 인식
    """
    wav_bytes_io.seek(0)
    fs, audio = wavfile.read(wav_bytes_io)
    # 다중 채널일 경우 첫 채널만 사용
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    # int16 -> float32 [-1, 1]
    audio = audio.astype(np.float32) / 32768.0

    result = model.transcribe(audio)
    return result

if __name__ == "__main__":
    # 사용 가능한 입력 장치 출력
    devices = list_input_devices()
    print("사용 가능한 입력 장치:")
    for idx, name in devices:
        print(f"{idx}: {name}")

    # 녹음할 장치 번호 지정 (리스트에서 마이크 장치 번호 선택)
    device_index = None  # 기본 장치 사용하려면 None, 아니면 숫자 지정 예) 2

    # 2가지 방법 모두 사용 가능

    # 방법 1: 기존 방식 - numpy 배열을 메모리로 넘겨서 인식
    model = whisper.load_model("base")

    audio_np = record_audio(duration_sec=5, fs=16000, device=device_index)
    wav_bytes = numpy_to_wav_bytes(audio_np, fs=16000)
    result = transcribe_from_memory(model, wav_bytes)
    print("=== 메모리 인식 결과 ===")
    print(result["text"])

    # 방법 2: WhisperSTT 클래스 활용 (파일 기반)
    # 녹음 데이터를 임시 파일로 저장 후 파일 경로로 인식
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(wav_bytes.read())
        tmp_path = tmp_file.name

    stt = WhisperSTT(model_name="base")
    text_from_class = stt.transcribe(tmp_path)
    print("=== WhisperSTT 클래스 인식 결과 ===")
    print(text_from_class)

    os.remove(tmp_path)
