# BERT_NER
- Create virtual environment  `virtualenv --python=python3.6 venv`
- Activate virtual environment `source venv/bin/activate`
- Install requirements.txt `pip install -r requirements.txt`

### Prediction App - 
Run `python app.py` in Terminal. It will create .csv of result with misspelt_name and list of ids for each misspelt city name.

- Test REST Api by python - 
```
import requests
out = requests.post('http://127.0.0.1:8000/get-entities', json={"sentence": "The Bush administration will ask Congress for more than $ 240 billion to cover the cost of military operations in Iraq and Afghanistan for the next two fiscal years . "})
out.text 
```