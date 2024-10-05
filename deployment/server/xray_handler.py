import io
from ts.torch_handler.base_handler import BaseHandler
from torchvision import transforms
from PIL import Image
from xray import CNNModel
import torch
import json
import os

#import logging

#logger = logging.getLogger(__name__)

class XRAYHandler(BaseHandler):

    def __init__(self):
        super(XRAYHandler, self).__init__()
        self.transform = transforms.Compose([    
            transforms.Resize((500, 500)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Rescale the pixel values
            ])

    def preprocess_one_image(self, req):
        """
        Process one single image.
        """
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")
         # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images


    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.
        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionaries with predictions and explanations is returned
        output = data.tolist()
        """
        output = data.tolist()
        return output
