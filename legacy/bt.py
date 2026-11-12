import bluetooth
import tts
import motor
 
phone = "34:80:B3:9A:56:E2"
 
def search():         
  devices = bluetooth.discover_devices(duration = 8, lookup_names = False)
  return devices

def unlock():    
    results = search()
    if not results:
      tts.voice('No keys found')
    for addr in results:
      if (addr == phone):
        tts.voice("Door Unlocked.")
        motor.open()
        break
      else:
        tts.voice("Key not found.")
