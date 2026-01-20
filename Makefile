.PHONY: setup clean test lint docker-build docker-run

IMAGE_NAME = social-signal-bench
TAG = latest

setup:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf outputs/temp/*

test:
	pytest tests/

lint:
	flake8 src scripts tests

format:
	black src scripts tests

docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) -f docker/Dockerfile .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs $(IMAGE_NAME):$(TAG)
