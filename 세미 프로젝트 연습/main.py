import streamlit as st
import tempfile
import os

from stt.whs_st import WhisperSTT

from summarizer.gpt_summarizer import GPTSummarizer
from video.test1 import save_wav_file

# 초기화
stt = WhisperSTT(model_name="base")
gpt = GPTSummarizer(api_key="YOUR_OPENAI_API_KEY")

st.title("시대별 음성 녹음 + Whisper STT + GPT 요약")

# 1. 사용자 음성 파일 업로드 또는 녹음
audio_file = st.file_uploader("음성 파일 업로드 (wav, mp3, m4a)", type=["wav","mp3","m4a"])

if audio_file:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # 2. Whisper STT 실행
    stt_text = stt.transcribe(tmp_path)
    st.subheader("Whisper 인식 결과")
    st.text_area("", value=stt_text, height=150)

    # 3. GPT 요약 실행
    summary = gpt.summarize(stt_text)
    st.subheader("GPT 요약 결과")
    st.text_area("", value=summary, height=150)

    # 임시 파일 삭제
    os.remove(tmp_path)
