# Bert Serve

A fast and ready-to-use sentence transformer model serving HTTP API using `bert.cpp`.
The model is `all-MiniLM-L12-v2` in the GGML format. The dimensions of the embedding are 384.

```shell
docker run -p 8080:8080 ghcr.io/llm-gitops/serve/bert-serve:v1.0.0

# Test the API
curl http://localhost:8080/v1/embeddings -d '{"input": "hello world"}'

# Count the number of dimensions in the embedding
curl -s http://localhost:8080/v1/embeddings -d '{"input": "hello world"}' | jq -r .data[0].embedding[] | wc -l
```
