#!/bin/bash

# Define image name
IMAGE_NAME="hemu33662/sentiment-analysis:latest"
PORT=8000

echo "Checking if a container is running on port $PORT..."

# Find the container ID using the port
CONTAINER_ID=$(docker ps -q --filter "publish=$PORT")

if [ ! -z "$CONTAINER_ID" ]; then
    echo "Stopping the existing container..."
    docker stop $CONTAINER_ID
    docker rm $CONTAINER_ID
else
    echo "No existing container running on port $PORT."
fi

echo "Checking for old images..."

# Find the old image ID
IMAGE_ID=$(docker images -q $IMAGE_NAME)

if [ ! -z "$IMAGE_ID" ]; then
    echo "Removing old image..."
    docker rmi -f $IMAGE_ID
else
    echo "No old image found."
fi

echo "Good to go for Deployment!"