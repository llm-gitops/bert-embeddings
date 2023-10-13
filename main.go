package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"sync"

	"github.com/gin-gonic/gin"
	bert "github.com/go-skynet/go-bert.cpp"
	"github.com/sashabaranov/go-openai"
)

var mutex sync.Mutex

func main() {
	var model string

	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	flags.StringVar(&model, "m", "./models/model.ggml", "path to a model file to load")

	err := flags.Parse(os.Args[1:])
	if err != nil {
		fmt.Printf("Parsing program arguments failed: %s", err)
		os.Exit(1)
	}
	llm, err := bert.New(model)
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		os.Exit(1)
	}
	fmt.Printf("Model loaded successfully.\n")
	defer llm.Free()

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.POST("/v1/embeddings", func(c *gin.Context) {
		mutex.Lock()         // Locks the processing of subsequent requests
		defer mutex.Unlock() // Ensures the lock will be released when processing is done

		req := openai.EmbeddingRequest{}
		err := c.BindJSON(&req)
		if err != nil {
			errResp := openai.ErrorResponse{
				Error: &openai.APIError{
					Code:           http.StatusBadRequest,
					Message:        err.Error(),
					Param:          nil,
					Type:           http.StatusText(http.StatusBadRequest),
					HTTPStatusCode: http.StatusBadRequest,
					InnerError:     nil,
				},
			}
			c.JSON(http.StatusBadRequest, errResp)
			return
		}
		// DEBUG
		fmt.Printf("Request: %+v\n", req)
		fmt.Printf("Input: %+v\n", req.Input)

		var inputTexts []string
		switch req.Input.(type) {
		case string:
			inputTexts = []string{req.Input.(string)}
		case []string:
			inputTexts = req.Input.([]string)
		default:
			errResp := openai.ErrorResponse{
				Error: &openai.APIError{
					Code:           http.StatusBadRequest,
					Message:        "input must be an array of string",
					Param:          nil,
					Type:           http.StatusText(http.StatusBadRequest),
					HTTPStatusCode: http.StatusBadRequest,
					InnerError:     nil,
				},
			}
			c.JSON(http.StatusBadRequest, errResp)
			return
		}

		data := make([]openai.Embedding, 0)
		for i, inputText := range inputTexts {
			embeddings, err := llm.Embeddings(inputText, bert.SetThreads(runtime.NumCPU()))
			if err != nil {
				errResp := openai.ErrorResponse{
					Error: &openai.APIError{
						Code:           http.StatusInternalServerError,
						Message:        err.Error(),
						Param:          nil,
						Type:           http.StatusText(http.StatusInternalServerError),
						HTTPStatusCode: http.StatusInternalServerError,
						InnerError:     nil,
					},
				}
				c.JSON(http.StatusInternalServerError, errResp)
				return
			}

			data = append(data, openai.Embedding{
				Object:    "embedding",
				Embedding: embeddings,
				Index:     i,
			})
		}

		resp := openai.EmbeddingResponse{
			Object: "list",
			Data:   data,
			Model:  0,
			Usage: openai.Usage{
				PromptTokens: 0,
				TotalTokens:  0,
			},
		}
		c.JSON(http.StatusOK, resp)
	})

	if err := r.Run(); err != nil {
		os.Exit(1)
	}
}
