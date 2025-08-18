from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Initialize the vLLM model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    yield

app = FastAPI(lifespan=lifespan)

class InferenceRequest(BaseModel):
    prompt: str
    temperature: float = 0.8
    max_tokens: int = 256

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        outputs = app.state.llm.generate(
            prompts=[request.prompt],
            sampling_params=sampling_params
        )
        generated_text = outputs[0].outputs[0].text
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)