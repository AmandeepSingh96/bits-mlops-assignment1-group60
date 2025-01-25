# bits-mlops-assignment1-group60

# ML Flow:

mflow ui

# Docker Commands used:

Requirements for Docker Environment: specified in requirements.txt file

docker build -t diabetes-prediction-app .

docker run -d -p 5000:5001 diabetes-prediction-app

//Push to Docker Hub:

docker tag diabetes-prediction-app amandeepsingh96/diabetes-prediction-app

docker push docker.io/amandeepsingh96/diabetes-prediction-app

https://hub.docker.com/repository/docker/amandeepsingh96/diabetes-prediction-app/general

# Execute Request (Postman):

POST - http://127.0.0.1:5000/predict

Content-Type: application/json

Body:
{
    "features": [90.0, 1.0, 1.0, 23.0, 94.0, 1.1, 0.167, 1.0, 2.0, 9.0]
}