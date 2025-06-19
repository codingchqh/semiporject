import streamlit as st
import os

from stt.whs_st import WhisperSTT
from summarizer.gpt_summarizer import GPTSummarizer
from video.test1 import save_wav_file
from video.test1 import make_video_from_text


import tempfile



# 1. Streamlit에서 음성 녹음 (업로드 또는 streamlit-webrtc 같은 패키지 활용 가능)
audio_file = st.file_uploader("음성 파일 업로드", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # 2. Whisper STT 실행
    stt = WhisperSTT()
    text = stt.transcribe(tmp_path)
    st.subheader("음성 인식 결과")
    st.write(text)

    # 3. GPT 요약 실행
    summarizer = GPTSummarizer(api_key="sk-로시작하는openai 키 입력")
    summary = summarizer.summarize(text)
    st.subheader("요약 결과")
    st.write(summary)

    # 4. 요약 텍스트로 영상 생성
    video_path = "output.mp4"
    make_video_from_text(summary, output_path=video_path)

    st.video(video_path)

    # 임시 파일 삭제
    os.remove(tmp_path)
