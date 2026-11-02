## Project overview
Example of integration of timestral model

# How to run
You need to have docker installed.

Create folders
```
mkdir -p {data,models,output}
```

Build docker file and run docker stack
```
docker compose up -d --build
```

Verify model is loaded
```
curl http://localhost:8000/healthz
```

Run evaluation
```
docker compose exec timestral python app/evaluate.py
```