# seq2seq date parser

A date parser using a sequence-to-sequence neural network.

## To train the model
`python train.py`

## To serve the website
```
export FLASK_APP=serve.py
flask run
```


## To use your new and improved model with the website

copy the model and vocabularies into `static` and update the paths in `serve.py`

