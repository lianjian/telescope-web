import os
import io, base64
import png
from PIL import Image

from . import telescope

from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.exceptions import abort

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
            SECRET_KEY='dev')

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Render landing page
    @app.route('/')
    def index():
        return render_template('index.html')


    @app.route('/demo/', methods=('GET', 'POST'))
    def demo():
        if request.method == 'POST':
            # Get image data from POST and open in memory for processing
            imageUrl = request.json['imgUrl']
            trimapUrl = request.json['trimapUrl']
            image = io.BytesIO(base64.b64decode(imageUrl.split(',')[1]))
            trimap = io.BytesIO(base64.b64decode(trimapUrl.split(',')[1]))
            
            # Generate alphamatte (numpy array)
            alphamatte = telescope.generateAlphamatte(image, trimap)
            
            # Convert alphamatte to PIL Image and store as BytesIO PNG
            alphamatteImage = Image.fromarray(alphamatte)
            byteIO = io.BytesIO()
            alphamatteImage.save(byteIO, format="PNG")
            alphamatteBuffer = byteIO.getvalue()
            byteIO.close()
            
            # Base64 encode image to send to <img> tag clientside
            alphamatteEncoded = base64.b64encode(alphamatteBuffer).decode('utf-8')
            
            return jsonify(result=alphamatteEncoded)

        return render_template('demo/index.html')

    return app
