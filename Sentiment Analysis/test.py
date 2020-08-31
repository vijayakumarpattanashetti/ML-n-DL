import requests

url = 'https://text-scorer.herokuapp.com/predict'
r = requests.post(url, json = 'loved it')
print(r.text.strip())
