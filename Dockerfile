FROM gcc:13

RUN apt-get update && apt-get install -y cmake

WORKDIR /workspace

COPY . .
RUN rm -rf go-bert.cpp || true
RUN git clone --recurse-submodules https://github.com/go-skynet/go-bert.cpp
RUN (cd go-bert.cpp && make libgobert.a)

FROM cgr.dev/chainguard/go:latest

COPY --from=0 /workspace /workspace
WORKDIR /workspace

RUN find . -name "lib*.a"
RUN go mod tidy && CGO_LDFLAGS="-L./go-bert.cpp" go build -o serve-bert main.go


FROM cgr.dev/chainguard/glibc-dynamic

WORKDIR /workspace
COPY --from=1 /workspace/serve-bert /workspace/serve-bert
COPY ./models/ggml-model-q4_1.bin /workspace/models/model.ggml

ENTRYPOINT ["/workspace/serve-bert"]
# CGO_LDFLAGS="-L./go-bert.cpp" go build -ldflags="-extldflags -static" -o main main.go
