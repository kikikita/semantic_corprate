"""Module for experiments"""
import os
import pyttsx3
import moviepy.editor as mp
from googletrans import Translator


async def text_to_video(text: str, img: str):
    """Experiment text to video"""
    engine = pyttsx3.init()
    engine.setProperty('voice', 'ru')
    engine.setProperty("rate", 260)
    engine.save_to_file(text, 'audio.mp3')
    engine.runAndWait()
    audio_clip = mp.AudioFileClip('audio.mp3')
    video_clip = mp.ImageClip(img).set_audio(audio_clip)\
        .set_duration(audio_clip.duration)
    video_clip.write_videofile("output.mp4", fps=24)
    os.remove("audio.mp3")


async def translate(text: str):
    """Experiment with translator"""
    translator = Translator()
    text_answer = translator.translate(text, dest='en').text
    return text_answer
