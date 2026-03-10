"""FastAPI REST API服务

提供 /chat 和 /feedback 端点，以及人格状态查询接口。
"""

import sys
from pathlib import Path

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..bootstrap import create_agent

logger = logging.getLogger(__name__)

# 全局Agent实例
_agent = None


def get_agent():
    """获取或创建全局Agent实例"""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


app = FastAPI(
    title="Chrysalis API",
    description="AI人格演化系统 REST API",
    version="0.1.0",
)


# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入消息", min_length=1)


class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent响应")
    interaction_id: str = Field("", description="交互记录ID")
    personality_summary: str = Field("", description="当前人格描述")


class FeedbackRequest(BaseModel):
    score: float = Field(..., ge=0.0, le=5.0, description="反馈评分 (0-5)")
    interaction_id: Optional[str] = Field(None, description="关联的交互ID")


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class PersonalityResponse(BaseModel):
    description: str
    traits: dict
    update_count: int
    interaction_count: int


class HistoryRecord(BaseModel):
    id: str = ""
    timestamp: str = ""
    user_input: str = ""
    agent_response: str = ""
    feedback: Optional[float] = None


# ==================== API端点 ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口"""
    try:
        agent = get_agent()
        response = agent.chat(request.message)

        personality_info = agent.get_personality_info()

        return ChatResponse(
            response=response,
            interaction_id=agent._last_interaction_id or "",
            personality_summary=personality_info["description"],
        )
    except Exception as e:
        logger.error("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """反馈接口"""
    try:
        agent = get_agent()
        normalized_score = request.score / 5.0
        agent.provide_feedback(normalized_score, request.interaction_id)

        return FeedbackResponse(
            success=True,
            message=f"反馈已记录: {request.score:.1f}/5.0",
        )
    except Exception as e:
        logger.error("Feedback error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/personality", response_model=PersonalityResponse)
async def get_personality():
    """获取当前人格状态"""
    try:
        agent = get_agent()
        info = agent.get_personality_info()

        return PersonalityResponse(
            description=info["description"],
            traits=info["traits"],
            update_count=info["update_count"],
            interaction_count=info["interaction_count"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(n: int = 10):
    """获取最近N条交互记录"""
    try:
        agent = get_agent()
        records = agent.memory.get_recent(n)

        return [
            HistoryRecord(
                id=r.get("id", ""),
                timestamp=r.get("timestamp", ""),
                user_input=r.get("user_input", ""),
                agent_response=r.get("agent_response", ""),
                feedback=r.get("feedback"),
            )
            for r in records
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "chrysalis"}


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """启动API服务器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
