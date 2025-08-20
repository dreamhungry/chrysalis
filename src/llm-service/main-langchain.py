from typing import Union
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import FastAPI
import langchain
from langchain.llms import VLLM
import time
import uvicorn

app = FastAPI()

llm = VLLM(model="Qwen/Qwen2.5-1.5B-Instruct",
           trust_remote_code=True,  # mandatory for hf models
           max_new_tokens=50,
           temperature=0.6
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/v1/gentext")
async def generate_text(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    print(prompt)
    output = llm(prompt)
    print("Generated text:", output)
    ret = {"text": output}
    return JSONResponse(ret)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)