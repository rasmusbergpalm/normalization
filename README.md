# seq2seq date parser

A date parser using a sequence-to-sequence neural network.

## To train the model

```
export FLASK_APP=serve.py
flask run
```

## To serve the website

`python serve.py`


## To use your new and improved model with the website

copy the model and vocabularies into `static` and update the paths in `serve.py`

