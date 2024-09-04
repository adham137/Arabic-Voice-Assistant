
import speech_recognition as sr
from gtts import gTTS
import IPython.display as ipd
from IPython.display import Audio
from pydub import AudioSegment
import pyaudio
import pygame



def arabic_speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please speak something in Arabic...")

        audio = recognizer.listen(source)

        try:

            text = recognizer.recognize_google(audio, language='ar-SA')
            print("أنت قلت:  ", text)
            return text

        except sr.UnknownValueError:
            text = "آسف، لم أتمكن من فهم الصوت."
            print(text)
            return text

        except sr.RequestError as e:
            print(f"Could not request results; {e}")

def text_to_speech(text, response_n):
    
    tts = gTTS(text, lang='ar', slow=False, tld="com.ar")
    audio_path = f"./responses/artext{response_n}.mp3"
    tts.save(audio_path)
    
    # ipd.Audio(audio_path, autoplay=True)

    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        continue
    

# Text = arabic_speech_to_text()
# print(Text)
# text_to_speech(Text)

