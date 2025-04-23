# Flask Deploy

A Flask application ready for Docker deployment.

## Docker Setup Instructions

### Building the Docker Image

```bash
docker build -t mulindwa/flask-deploy .
```

### Running the Docker Container Locally

```bash
docker run -p 5000:5000 mulindwa/flask-deploy
```

Or using Docker Compose:

```bash
docker-compose up
```

### Pushing to Docker Hub

1. Log in to Docker Hub:
   ```bash
   docker login
   ```

2. Push your image:
   ```bash
   docker push mulindwa/flask-deploy
   ```

### Pulling and Running from Docker Hub

```bash
docker pull mulindwa/flask-deploy
docker run -p 5000:5000 mulindwa/flask-deploy
```

## Note

Make sure to replace `mulindwa` with your actual Docker Hub username.
