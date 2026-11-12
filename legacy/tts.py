from gtts import gTTS
import os
import pyttsx

def voice(words):
    tts = gTTS(text=words, lang='en-uk')
    tts.save("voice.mp3")
    os.system("mpg123 voice.mp3")
def voice1(word):   
    engine = pyttsx.init()
    rate=engine.getProperty('rate')
    engine.setProperty('rate',rate-10)
    engine.setProperty('voice', 'hindi')
    engine.say(word)
    engine.runAndWait()    
