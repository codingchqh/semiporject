from moviepy.editor import TextClip, CompositeVideoClip
import numpy as np
from scipy.io import wavfile

def make_video_from_text(text: str, output_path="output.mp4", duration=10):
    clip = TextClip(text, fontsize=32, color='white', size=(1280, 720), font='Malgun Gothic', bg_color='black')
    clip = clip.set_duration(duration)
    video = CompositeVideoClip([clip])
    video.write_videofile(output_path, fps=24)

def save_wav_file(path, audio_np, fs=16000):
    wavfile.write(path, fs, audio_np)
