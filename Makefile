IMAGE_NAME=rag-python

build:
	docker build -t $(IMAGE_NAME) .

run_insertion:
	docker run --rm -network=host -v $(shell pwd):/app $(IMAGE_NAME) python /app/data-insertion.py

run_query:
	docker run --rm -network=host -v $(shell pwd):/app $(IMAGE_NAME) python /app/data-query.py