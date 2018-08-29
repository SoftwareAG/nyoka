from requests.auth import HTTPBasicAuth
import requests
import json


class ZementisCalls():

    def authentication(self):
        self.url = "http://dcindgo01:8083/adapars/"
        self.auth = HTTPBasicAuth('Administrator', 'manage')
        response = requests.get(self.url,auth=self.auth)
        return response.ok

    def check_model_existence(self,model):
        # Check if the model is already uploaded into Zementis, if yes delete it
        response = requests.get(self.url + "models/", auth=self.auth)
        if model in json.loads(response.text)['models']:
            requests.delete(self.url + "model/"+model ,auth=self.auth)

        return response.ok

    def pmml_upload(self,pmml_file):
        # Upload the PMML into Zementis
        pmml_f = open(pmml_file,"r")
        pmml = {'file': pmml_f}

        self.pmml_uploads = requests.post(self.url + "models/", files=pmml, auth=self.auth)

        return self.pmml_uploads.status_code


    def extract_model_name(self):
        # Extract the model name from model properties
        model_properties = json.loads(self.pmml_uploads.text)
        self.model_name = model_properties['modelName']
        return self.model_name

    def get_predictions(self,test_csv,modelname):
        # Perform the predictions using test data
        test_file = open(test_csv,"r")
        test_csv = {'file':test_file}
        predictions_data = requests.post(self.url + "apply/" + modelname, files=test_csv, auth=self.auth)
        predictions = predictions_data.text.split()
        predictions.pop(0)
        predictions = [predictions[i].split(",")[0] for i in range(len(predictions))]
        return predictions