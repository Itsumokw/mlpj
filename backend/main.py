from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import (
    PreprocessRequest, PreprocessResponse, 
    TrainRequest, TrainResponse, 
    TrainingStatus, ForecastRequest, ForecastResponse
)
from data_processor import process_data
from trainer import ModelTrainer
from predictor import ForecastPredictor
import uuid
import threading

app = FastAPI(
    title="Time Series Forecasting API",
    description="Backend API for time series forecasting lab",
    version="1.0.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
trainer = ModelTrainer()
predictor = ForecastPredictor()

# 用于存储预处理结果
preprocessed_data_store = {}

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(request: PreprocessRequest):
    """数据预处理端点"""
    try:
        # 处理数据
        response = process_data(request)
        # 存储预处理结果（在实际应用中应使用数据库）
        job_id = str(uuid.uuid4())
        preprocessed_data_store[job_id] = response.dict()
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """训练模型端点"""
    try:
        # 生成唯一的job_id
        job_id = str(uuid.uuid4())
        
        # 启动后台训练任务
        background_tasks.add_task(
            trainer.train_model,
            job_id,
            request.config.dict(),
            request.train_data
        )
        
        return TrainResponse(job_id=job_id, status="queued")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=TrainingStatus)
async def get_status(job_id: str):
    """获取训练状态端点"""
    status = trainer.get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    status["job_id"] = job_id
    return status

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """预测端点"""
    
    result = predictor.forecast(request.dict())
    return result
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)