FROM pytorch/torchserve:latest

WORKDIR /api

COPY . .

# Create the model store directory
RUN mkdir -p model_store

# Create the .mar file for the model
RUN torch-model-archiver --model-name xray_model --version 2.0 --serialized-file xray_cnn.pt --model-file xray.py --handler xray_handler.py --extra-files index_to_name.json -f

# Move the .mar file to the model store directory
RUN mv xray_model.mar model_store/

# Start TorchServe
CMD ["torchserve", "--start", "--ncs", "--model-store", "model_store", "--models", "xray=xray_model.mar", "--foreground", "--ts-config", "config.properties", "--disable-token-auth"]
