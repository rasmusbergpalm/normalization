import numpy as np
from flask import Flask
from flask import send_from_directory
from keras.models import load_model
from flask import request

import data

print("Loading data...")
reader = data.ParallelReader("static/empty.txt", "static/source-vocab.txt", "", "static/empty.txt", "static/target-vocab.txt", "")

print("Loading model...")
model = load_model('static/model.01-0.02.hdf5', {'all_acc': lambda x, y: x})

print("Starting server...")
app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/about")
def about():
    return send_from_directory("static", "about.html")


@app.route("/parse")
def parse():
    source_str = request.args.get('q')
    source_idx = reader.source.str_to_idx(source_str)
    output_idx = model.predict_on_batch(np.array([source_idx], dtype=np.int32))

    output = reader.target.idx_to_str(np.argmax(output_idx, -1))
    return output[0]


if __name__ == "__main__":
    app.run(host='0.0.0.0')
