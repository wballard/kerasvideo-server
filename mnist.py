"""
This module serves as the API provider for MNIST digit processing.
"""

from PIL import Image
from PIL.ImageOps import fit, grayscale
import io
import numpy as np
import json


def post_image(file):
    """
    Given a posted image, classify it using the pretrained model.

    This will take 'any size' image, and scale it down to 28x28 like our MNIST
    training data -- and convert to grayscale.

    Parameters
    ----------
    file:
        Bytestring contents of the uploaded file. This will be in an image file format.
    """
    image = Image.open(io.BytesIO(file.read()))
    image = grayscale(fit(image, (28, 28)))
    image_bytes = image.tobytes()
    image_array = np.reshape(np.frombuffer(image_bytes,  dtype=np.uint8), (28, 28))
    print(image_array.shape)
    return json.dumps({'digit': None})

