import requests
import json
from requests.auth import HTTPBasicAuth
import datetime

urlAlarm = "https://ai.cumulocity.com/alarm/alarms"

headersAlarm = {
'Content-Type': "application/vnd.com.nsn.cumulocity.alarm+json",
'Accept': "application/vnd.com.nsn.cumulocity.alarm+json",
}

payload_Critical = {
    "source": {
        "id": "143428" },
    "type": "Drink",
    "text": "There has been an error while painting",
    "severity": "MAJOR",
    "status": "ACTIVE",
    "time": "2014-03-03T12:03:27.845Z"
}
          
response = requests.request("POST", urlAlarm, data=json.dumps(payload_Critical),
                headers=headersAlarm,auth=HTTPBasicAuth('Rainer.Burkhardt@softwareag.com', 'cum 2418 Point Loma'))

    