import RPi.GPIO as GPIO
import time
import urllib2
import tts
import wiki
import bt
import os
import weather
import random
import joke

GPIO.setmode(GPIO.BOARD)
GPIO.setup(12, GPIO.OUT)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)

def no_rep():
    nr=['Let me be sure I understand which information you are looking for.','Thats a timely question, because I am currently gathering information on that.','This is scary for me to say, so I am hoping you can really hear me and try not to judge me or give me any advice.',' I will totally take a look at that.']
    n=random.randint(0,3)
    tts.voice(nr[n])

def wish_night():
    wn=['Do you think about me before you go to sleep? Good night.','I just wanted to let you know I think of you. Good Night','I am already missing you. Good night']
    b=random.randint(0,2)
    tts.voice(wn[b])

def wish_morning():
    wm=['Good thoughts precede great deeds. Great deeds precede success. Have a great day.','This is not just another day, this is yet another chance to make your dreams come true. Good morning.']
    a=random.randint(0,1)
    tts.voice(wm[a])

def edit():
    os.system('echo "NULL" > /var/www/html/test/val.txt')

while True:
    with open('/var/www/html/test/val.txt') as data_file:
        val= data_file.readline()
    val=val.upper()
    if 'LIGHTS' in val:
        if 'BEDROOM' in val:
            if 'ON' in val:
                GPIO.output(11,True)
                tts.voice("Bedroom Lights on.")
                edit()
                
            elif 'OFF' in val:
                GPIO.output(11,False)
                tts.voice("Bedroom Lights off.")
                edit()
                
        if ('LIVING' in val and 'ROOM' in val):
            if 'ON' in val:
                GPIO.output(12,True)
                tts.voice("Living room Lights on.")
                edit()
                
            elif 'OFF' in val:
                GPIO.output(12,False)
                tts.voice("Living room Lights off.")
                edit()
                
        
    elif 'WIKIPEDIA' in val:
        wiki.search(val[10: ])
        edit()

    elif ('UNLOCK' in val and 'DOOR' in val):
        bt.unlock()
        edit()

    elif 'WEATHER' in val:
        weather.weather()
        edit()

    elif 'TEMPERATURE' in val:
        weather.temp()
        edit()

    elif ('GOOD'in val) and ('NIGHT' in val):
        GPIO.output(12,False)
        wish_night()
        tts.voice('Sleep mode on.')
        edit()

    elif ('GOOD'in val) and ('MORNING' in val):
        wish_morning()
        tts.voice('You got'+str(random.randint(0,2))+'tasks today')
        edit() 

    elif ('HEY' in val) or ('HELLO' in val):
        tts.voice('Hello, how may I help you?')
        edit()

    elif ('BYE' in val) and ('LUCY' in val):
        tts.voice('have a great day. Sleep mode on.')
        GPIO.output(12,False)
        edit()

    elif 'JOKE' in val:
        joke.joke()
        edit()

    
