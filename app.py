from flask import Flask, render_template,request,jsonify
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
from mlProject import logger
from flask_cors import CORS

app = Flask(__name__) # initializing a flask app
CORS(app)



@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            # Get JSON data from the request
            # data = request.get_json()

            logger.info("before")
            brand = request.form['brand']
            spec_rating = int(request.form['spec_rating'])
            Ram = int(request.form['Ram'])
            ROM = int(request.form['ROM'])
            ROM_type = int(request.form['ROM_type'])
            display_size = float(request.form['display_size'])
            resolution_width = float(request.form['resolution_width'])
            resolution_height = float(request.form['resolution_height'])
            OS = request.form['OS']
            warranty = int(request.form['warranty'])
            gpu_type = request.form['gpu_type']
            cpu_core = int(request.form['cpu_core'])
            cpu_threads = int(request.form['cpu_threads'])
            processor_brand = request.form['processor_brand']
            processor_gen = int(request.form['processor_gen'])
            processor_version = request.form['processor_version']
            logger.info("after")


            data = [brand,spec_rating,Ram,ROM,ROM_type,display_size,resolution_width,resolution_height,OS,
                    warranty,gpu_type,cpu_core,cpu_threads,processor_brand,processor_gen,processor_version]
            
            print(data)
            obj = PredictionPipeline()
            predicted = int(obj.predict(data)[0])
            print(predicted)


            return render_template('result.html', prediction = str(predicted))
            # print(predicted)
            # return jsonify({"success":True,"predicted":predicted})

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)