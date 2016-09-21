import logging
import json, random, urllib, cStringIO

from flask import render_template, request, redirect, url_for, session, jsonify
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename

from binascii import a2b_base64
from PIL import Image
from StringIO import StringIO
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import model_from_json
import numpy as np, requests, os, cv2

from . import app

logger = logging.getLogger('app')

# load in keras model
with open('/Users/AlexH/project5/image_app/app/static/model.json', 'r') as f:
	loaded_model_json = f.read()
model = model_from_json(loaded_model_json)

# load weights into model
model.load_weights("/Users/AlexH/project5/image_app/app/static/model.h5")
print("Loaded model from disk")

class InputForm(Form):
    url = fields.StringField('url', validators=[DataRequired()])
    submit = fields.SubmitField('Submit')

@app.route('/index_webcam.html', methods=['GET', 'POST'])
def index_webcam():
    if request.method == 'POST':
        r = request.get_json()
        url = r['url'] + "="
        url = url[22:]

        path = "app/static/temp.jpg"
        binary_data = a2b_base64(url)
        with open(path, 'wb') as f:
            f.write(binary_data)
        
        return jsonify({"result": "hello cat"})

    return render_template('index_webcam.html')

@app.route('/index_link.html', methods=['GET', 'POST'])
def index_link():
    form = InputForm()
    
    if form.validate_on_submit():
        url = form.url.data
        session['url'] = url
        return redirect(url_for('predict', mode='url', url=url))

    return render_template('index_link.html', form=form)

@app.route('/index_upload.html', methods=['GET', 'POST'])
def index_upload():
    if request.method == 'POST':
        f = request.files['file']
        path = "app/static/temp.jpg"

        if os.path.isfile(path):
            os.remove(path)
        f.save(path)

        return redirect(url_for('predict', mode='upload'))

    return render_template('index_upload.html')



# @app.route('/webcam', methods=['GET', 'POST'])
# def take_pic():
#     if request.method == 'POST':
#         r = request.get_json()
#         #import pdb; pdb.set_trace()
#         #print r
#         url = r['url'] + "="
#         #print url
#         #print json.loads(r)
#         #url = json.loads(r)['url']
#         url = url[22:]

#         path = "app/static/temp.jpg"
#         binary_data = a2b_base64(url)
#         with open(path, 'wb') as f:
#             f.write(binary_data)
        
#         print "i'm in the take_pic() function"
#         return jsonify({"result": "hello world"})
#         #return redirect(url_for("predict", mode="upload"))

#     return redirect(url_for("index"))

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         path = "app/static/temp.jpg"
        
#         if os.path.isfile(path):
#             os.remove(path)
#         f.save(path)
        
#         return redirect(url_for('predict', mode='upload'))
    
#     return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = InputForm()
    up = False
    own = False
    data = []  
    actors = ['Angelina Jolie','Christian Bale','Emma Stone','Harrison Ford','Jennifer Lawrence','Keanu Reeves','Leonardo Dicaprio',
             'Natalie Portman','Scarlett Johannson','Tom Cruise']
    
    mode = request.args['mode']

    if mode == 'url':
        try:
            url = request.args['url']
            r = requests.get(url)
            im = np.array(Image.open(StringIO(r.content)))
        except:
            print "No Image!"
    elif mode == 'upload':
        try:           
            url = "app/static/temp.jpg"
            im = np.array(Image.open(url))
            up = True
        except:
            print "No image!"
    elif mode == 'upload_cam':
        try:
            url = "app/static/temp.jpg"
            im = np.array(Image.open(url))
            if im.shape[2] == 4:
                im = np.delete(im, 3, axis=2)
            own = True
        except:
            print "No image!"

    try:
	    # read in image
        im = cv2.resize(im, (224,224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)

	    # make prediction
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        out = model.predict(im)[0]

        for i in range(10):
            data.append({'image': str(i), 'name': actors[i], 'prob': out[i]})
            data = sorted(data, key=lambda x: x['prob'], reverse=True)[:3]

        for i in data:
            i['prob'] = str(int(i['prob']*100)) + "%"
    except:
        data = []
        url = None

    random_qstr = "?" + str(random.randint(1,10000))
    return render_template('index.html', form=form, url=url, random_qstr=random_qstr, own=own, up=up, data=data)

@app.route('/', methods=('GET', 'POST'))
def index():
    form = InputForm()
    
    # check if "Clear" was called
    if form.data['url'] == "":
        print "blank!"
    
    if form.validate_on_submit():
        url = form.url.data
        session['url'] = url
        return redirect(url_for('predict', mode='url', url=url))
    
    return render_template('index.html', form=form)
