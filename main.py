from fastapi import FastAPI, Query
import uvicorn
from pydantic import BaseModel
from inference import setup_pipeline


pipeline = setup_pipeline(
    model_name="bert-base-cased", 
    # model_path="/Users/alexuvarovskiy/Documents/course_competition/trainer/checkpoint-1140/pytorch_model.bin"
    model_path="/app/trainer/checkpoint-1140/pytorch_model.bin"
)

app = FastAPI(root_path="/api/v1")


class ScoredTextOut(BaseModel):
    score: float


@app.get("/regression", response_model=ScoredTextOut)
async def read_classification(text: str = Query(..., min_length=1)):
    return pipeline(text.lower())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
