build:
	docker build -t ghcr.io/llm-gitops/serve/bert-serve:v1.0.0 .
	docker push ghcr.io/llm-gitops/serve/bert-serve:v1.0.0
