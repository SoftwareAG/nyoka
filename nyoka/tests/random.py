import json

def start():
    data = json.load(open("nyoka_downloads.json",'r'))
    print(data)

if __name__ == "__main__":
    start()