
from flask_app import app
from flask import render_template, request, jsonify, redirect, flash

     
    
@app.route('/')
@app.route('/index.html')
def index():
    return app.send_static_file('index.html')

@app.route('/prediction', methods = ['POST'])
def predict():
    print('prediction request')
    if 'file' not in request.files:            
            flash('No se selecciono ninguna imagen'
            print('No image selected')
            return redirect(request.url)

    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':        
        flash('No se selecciono ninguna imagen')
        print('No image name')
        return redirect(request.url)

    if file and app.allowed_file(file.filename):
        
        X = app.get_input_tensor(file)
           
        cnn_prediction = app.model.predict(X)
        print('prediction: ', cnn_prediction)
        label = app.get_label_sparse(cnn_prediction)       
        #label = app.get_label_categorical(cnn_prediction)
        print(label)     
        prediction = {'prediction' : str(cnn_prediction), 'result' : label}
        return jsonify(prediction)
    else:
        print('Not allowed file')