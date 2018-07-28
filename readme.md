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

Open your browser to (http://localhost:5000/ui).

# Quick Start Docker
```
docker build --tag kerasvideo-server .
docker run -p 5000:5000 kerasvideo-server
```

Open your browser to (http://localhost:5000/ui).

## Models
Models are pretrained and saved individually, and then served at REST API endpoints. You will need a 
model trained in order for the server to process!

### `train_mnist.py`
Classification models are provided for MNIST digits, which creates a saved model file resulting
from Keras training. This trained models is then used as the classification function.

## `server.py`
The server is a python script
API endpoints handle posted files, and conversion to the appropriate vector encoding for use with
keras. Once a `POST` image is encoded, it is sent to the loaded model for classification. Once classified,
the classification results are serialized to JSON and returned.

## Docker
The included Dockerfile will create a container, complete with the REST server -- and pretrained models. Distributing
your models to your running servers in practice is a mildly painful exercise, so packing the binary data of the
trained model into a Docker container eases deployment.

With this Docker based approach, the server and model are completely self contained.