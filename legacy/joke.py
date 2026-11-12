import urllib2
import json
import tts

def joke():
    response = urllib2.urlopen("http://tambal.azurewebsites.net/joke/random")
    data= json.load(response)
    tts.voice(data['joke'])

