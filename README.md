# X-Ray Chest Classification System

A deep learning-based chest X-ray classification system with microservice architecture

## Project Description
This project provides an intelligent system for classifying chest X-ray images into "normal" and "opacity" categories. The system uses a custom CNN model implemented with PyTorch and is served through TorchServe in a microservice architecture.

## Key Features
- Binary classification of chest X-ray images
- Image preprocessing and data augmentation for improved model performance
- Web interface for image upload and classification
- Microservice architecture using Docker
- Scalable and easy deployment

## Project Structure
```
full-stack-xray-classification/
├── model/
│   └── xrayChest_cnn.ipynb      # Model training notebook
├── deployment/
│   ├── server/                  # TorchServe server
│   ├── client/                  # Web interface
│   └── docker-compose.yaml      # Docker configuration
├── requirements.txt             # Project dependencies
└── README.md
```

## Prerequisites
- Python 3.8+
- Docker and Docker Compose
- CUDA (for GPU training - optional)

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/full-stack-xray-classification.git
cd full-stack-xray-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the system with Docker:
```bash
cd deployment
docker-compose up --build
```

4. Access the interfaces:
- Web UI: http://localhost:9696
- TorchServe API: http://localhost:8080

## Technical Details

### Deep Learning Model
- Custom Convolutional Neural Network (CNN)
- Image preprocessing including resizing, normalization, and contrast enhancement
- Data augmentation techniques for better generalization

### System Architecture
- Frontend: Web interface for upload and result display
- Backend: TorchServe for model serving
- Containerization: Docker for service management and deployment

## Contributing
Your suggestions and contributions to improve the project are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Submit a Pull Request

## License
This project is licensed under the MIT License.