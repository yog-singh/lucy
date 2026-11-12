import urllib
import xml.etree.ElementTree as ET
from pprint import pprint
import tts 


def weather():
    url="http://api.openweathermap.org/data/2.5/weather?lat=22.57&lon=88.36&appid=ddc80cbc49644abd43d989deb781d0e3&mode=xml"
    response=urllib.urlopen(url)
    data= ET.parse(response)
    root=data.getroot()
    temp=root[1].attrib
    humid=root[2].attrib
    cloud=root[5].attrib
    temp_val=float(temp['value'])-273
    humid_val=int(humid['value'])
    cloud_type=cloud['name']
    text="The temperature is "+str(int(temp_val))+" degree Celsius "+str(humid_val)+'percent humidity and'+cloud_type
    tts.voice(text)

def temp():
    url="http://api.openweathermap.org/data/2.5/weather?lat=22.57&lon=88.36&appid=ddc80cbc49644abd43d989deb781d0e3&mode=xml"
    response=urllib.urlopen(url)
    data= ET.parse(response)
    root=data.getroot()
    temp=root[1].attrib
    temp_val=float(temp['value'])-273
    text="The temperature is "+str(int(temp_val))+" degree Celsius"
    tts.voice(text)
