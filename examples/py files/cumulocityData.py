import requests
import json
from requests.auth import HTTPBasicAuth
import datetime
import time
import random

url2 = "https://ai.cumulocity.com/measurement/measurements"

for i in range(0,10000):
    tt=datetime.datetime.now()
    payload2={'type': 'Sensor1',
     'time': str(tt.date())+'T'+str(tt.hour)+':'+str(tt.minute)+':'+str(tt.second)+'+05:30',
     'source': {'id': "143428"},
     'temperature': {'temperature':{'value': random.randint(1,100)}},
     'pressure': {'pressure':{'value': random.randint(101,200)}}}
    #print (i,' >>>>>> ',payload2['time'],payload2['Sensor1'])
    headers = {
    'Content-Type': "application/json",
    'Accept': "application/vnd.com.nsn.cumulocity.measurement+json",
    'cache-control': "no-cache",
    'Postman-Token': "2d5fa27d-c8c8-428c-b2f9-0efe9490b716"
    }
    response = requests.request("POST", url2, data=json.dumps(payload2), 
                            headers=headers,auth=HTTPBasicAuth('Rainer.Burkhardt@softwareag.com', 'cum 2418 Point Loma'))

