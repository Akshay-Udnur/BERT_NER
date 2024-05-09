from flask import Flask, request
from prediction import *

app = Flask(__name__)

@app.route('/test', methods = ['GET'])
def test_server():
    return "server is running."

@app.route('/get-entities', methods = ['GET', 'POST']) 
def sent_result():
    sentence = request.get_json()['sentence']
    return str(main(sentence))

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port = 8000)