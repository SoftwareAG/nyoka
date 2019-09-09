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
        self.endpoint = os.environ['ADAPA_URL']
        self.username = os.environ['ADAPA_UN']
        self.password = os.environ['ADAPA_PW']

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

    def score_single_record(self, model_name):
        res = requests.get(self.endpoint+"apply/"+model_name, auth = HTTPBasicAuth(self.username, self.password))
        return res.json()['outputs'][0]

    def score_in_zserver(self, model_name, test_file, model_type=None):
        mode = 'r' if test_file.endswith(".csv") else 'rb'
        files = {'file': open(test_file,mode)}
        res = requests.post(self.endpoint+"apply/"+model_name, auth = HTTPBasicAuth(self.username, self.password),files=files)
        if model_type:
            if model_type=='DN':
                if test_file.endswith(".csv"):
                    info=res.text.split("\n")[1]
                    info = info.replace('"','')
                    predictions = info.split(",")[0]
                    probabilities = {out.split(":")[0]:float(out.split(":")[1]) for out in info.split(",")[2:]}
                else:
                    resp=json.loads(res.text)
                    outs = resp["outputs"][0]
                    predictions = outs["predicted_label"]
                    probabilities = {out.split(":")[0]:float(out.split(":")[1]) for out in outs["top5_prob"].split(",")}
            elif model_type=='RN':
                results = res.json()['outputs'][0]['predicted_LabelBoxScore']
                results = json.loads(results)
                boxes = []
                scores = []
                labels = []
                for res_ in results:
                    labels.append(res_[0])
                    scores.append(res_[1])
                    boxes.append(res_[2:])
                return boxes, scores, labels
            else:
                result = float(res.text.split('\n')[1])
                return result
        else:
            all_rows = res.text.split('\n')
            predictions = []
            probabilities = []
            if all_rows[0].split(",").__len__() == 1:
                for row in all_rows[1:]:
                    if row.__class__.__name__ == 'str' and row.__len__() > 0:
                        row = ast.literal_eval(row)
                    else:
                        continue
                    predictions.append(row)
                predictions = numpy.array(predictions)
                probabilities = None
            else:
                for row in all_rows[1:]:
                    cols = row.split(",")
                    if cols[0] == '':
                        continue
                    predictions.append(ast.literal_eval(cols[-1]))
                    probabilities.append(numpy.array([ast.literal_eval(c) for c in cols[:-1]]))
                predictions = numpy.array(predictions)
        return predictions, probabilities
        
    def compare_predictions(self, z_pred, m_pred):

        if z_pred[0].__class__.__name__ == 'str':
            for z, m in zip(z_pred, m_pred):
                if z != m:
                    return False
        else:
            for z, m in zip(z_pred, m_pred):
                if "{:.3f}".format(z) != "{:.3f}".format(m):
                    return False
        return True

    def compare_probability(self, z_prob, m_prob):
        for z, m in zip(z_prob, m_prob):
            for z_, m_ in zip(z,m):
                if "{:.3f}".format(z_) != "{:.3f}".format(m_):
                    return False
        return True

        
