import bluetooth

phone = "34:80:B3:9A:56:E2"
 
def search():         
  devices = bluetooth.discover_devices(duration = 5,flush_cache=True, lookup_names = False)
  return devices

def unlock():    
    results = search()
    for addr in results:
      if addr == phone:
        print 'Found'
      else:
        print' Key not found'
def edit():
    os.system('echo "new"> /var/www/html/test/val.txt')
with open('/var/www/html/test/val.txt') as dfile:
  data= dfile.read()
print data
