import wikipedia
import tts
    
def search(query):
     try:
          result= wikipedia.summary(query, sentences=1)
          tts.voice(result)
     except wikipedia.exceptions.DisambiguationError as de:
          error= de.options
          print (error[0])
     except wikipedia.exceptions.PageError as pe:
          error= pe.error
          tts.voice("Nothing found on Wikipedia.")
