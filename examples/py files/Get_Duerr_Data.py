import requests
from requests.auth import HTTPBasicAuth

url = "https://banduerr.eur.ad.sag/measurement/measurements/series"

querystring = {"dateFrom":"2019-05-24T10:38:00%2B05:30",
               "dateTo":"2019-05-24T11:40:00%2B05:30",
               "pageSize":"1440","revert":"true",
               "series":"PCU1~AHS1~SetValueVoltage.A",
               "source":"5832504"}

headers = {
    'Authorization': "Basic ZWRnZS9kdWVycl9hZG1pbjpwYXNzd29yZDMyMjA=",
    'User-Agent': "PostmanRuntime/7.13.0",
    'Accept': "*/*",
    'Cache-Control': "no-cache",
    'Postman-Token': "2a19e049-7416-4e46-8ac7-860c045d0c02,aac5d417-9952-4947-abc6-01d9a49f45b3",
    'Host': "banduerr.eur.ad.sag",
    'accept-encoding': "gzip, deflate",
    'Connection': "keep-alive",
    'cache-control': "no-cache"
    }

q2= 'dateFrom=2019-05-24T10:38:00%2B05:30&dateTo=2019-05-24T11:40:00%2B05:30&pageSize=1440&revert=true&series=PCU1~AHS1~SetValueVoltage.A&source=5832504'

response = requests.request("GET", url,params=q2,verify=False,
                            headers=headers,auth=HTTPBasicAuth('edge/duerr_admin', 'password3220'))

print(response.text)