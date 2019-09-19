import os
import requests
from requests.auth import HTTPBasicAuth

class SetupServer:

    def __init__(self):
        self.endpoint = os.environ['DOCKER_ADAPA_URL']
        self.username = os.environ['DOCKER_ADAPA_UN']
        self.password = os.environ['DOCKER_ADAPA_PW']

    def upload_license(self, file_name):
        files = {'file': open(file_name,'r')}
        res = requests.post(self.endpoint+"license", auth = HTTPBasicAuth(self.username, self.password),files=files)
        print(res.json())

    def upload_resource(self, file_name):
        files = {'file': open(file_name,'rb')}
        res = requests.post(self.endpoint+"resource", auth = HTTPBasicAuth(self.username, self.password),files=files)
        print(res.json())

if __name__ == "__main__":
    setup = SetupServer()
    setup.upload_license("zementis.license")
    setup.upload_resource("CNN.jar")
    setup.upload_resource("KerasRetinaNet.jar")