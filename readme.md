# Overview
This is a REST Server for Keras models, utilizing OpenAPI to serve image classification
models.

Input images are `POST` to the served APIs, and classification `JSON` results are returned.

# Quick Start
```
pip install -r requirements.txt
python train_mnist.py
python server.py
```

Open your browser to (http://localhost:4000/ui).


## Models
Models are pretrained and saved individually, and then served at REST API endpoints.

### `train_mnist.py`
Classification models are provided for MNIST digits, which creates a saved model file resulting
from Keras training. This trained models is then used as the classification function.

## `server.py`
The server is a python script
API endpoints handle posted files, and conversion to the appropriate vector encoding for use with
keras. Once a `POST` image is encoded, it is sent to the loaded model for classification. Once classified,
the classification results are serialized to JSON and returned.

## Utilities
`dump_mnist.py` is provided to create a set of image files from the encoded MNIST digit dataset in order to
exercise the post API.