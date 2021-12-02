from  flask import Flask,request,jsonify
import random
from Key_wordspoting_service import Keyword_spotting_service
import os
app = Flask(__name__)

@app.route("/predict",methods=["POST"])

def predict():
    audio_file =  request.files["file"]
    file_name = str(random.randint(0,100000))
    audio_file.save(file_name)

    kss = Keyword_spotting_service()
    predicted_keyword = kss.predict(file_name)
    os.remove(file_name)
    data = {"keyword":predicted_keyword}
    return jsonify(data)

if __name__ =="__main__":
    app.run(debug=False)

