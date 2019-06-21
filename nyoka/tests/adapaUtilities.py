import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import requests
import json
from requests.auth import HTTPBasicAuth
import ast
import numpy

class AdapaUtility:

    def __init__(self):
        self.endpoint = os.environ["ADAPA_URL"]
        self.username = os.environ["ADAPA_UN"]
        self.password = os.environ["ADAPA_PW"]

    def upload_to_zserver(self, file_name):
        files = {'file': open(file_name,'r')}
        res = requests.post(self.endpoint+"model", auth = HTTPBasicAuth(self.username, self.password),files=files)
        if res.status_code == 409:
            model_name = res.json()['errors'][0].split("\'")[1]
            status_code = self.delete_model(model_name)
            if status_code != 200:
                print("Something went wrong! Staus code ",status_code)
                return
            files = {'file': open(file_name,'r')}
            res = requests.post(self.endpoint+"model", auth = HTTPBasicAuth(self.username, self.password),files=files)
        return res.json()['modelName']

    def delete_model(self, model_name):
        res = requests.delete(self.endpoint+"model/"+model_name, auth=HTTPBasicAuth(self.username,self.password))
        return res.status_code

    def score_in_zserver(self, model_name, test_file):
        files = {'file': open(test_file,'r')}
        res = requests.post(self.endpoint+"apply/"+model_name, auth = HTTPBasicAuth(self.username, self.password),files=files)
        all_rows = res.text.split('\r\n')
        predictions = []
        probabilities = []
        if all_rows[0].split(",").__len__() == 1:
            for row in all_rows[1:]:
                if row.__class__.__name__ == 'str' and row.__len__() > 0:
                    row = ast.literal_eval(row)
                else:
                    continue
                predictions.append(row)
            probabilities = None
        else:
            for row in all_rows[1:]:
                cols = row.split(",")
                if cols[0] == '':
                    continue
                predictions.append(ast.literal_eval(cols[-1]))
                probabilities.append([ast.literal_eval(c) for c in cols[:-1]])
        return predictions, probabilities
        
    def compare_predictions(self, z_pred, m_pred):
        count = 0
        for z, m in zip(z_pred, m_pred):
            if not numpy.allclose(numpy.array(z), numpy.array(m),1,0):
                count += 1
        return count

    def compare_probability(self, z_prob, m_prob):
        count = 0
        for z, m in zip(z_prob, m_prob):
            if not numpy.allclose(numpy.array(z), numpy.array(m),1,0):
                count += 1
        return count

        