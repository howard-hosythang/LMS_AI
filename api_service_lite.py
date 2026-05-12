"""Lightweight FastAPI service for documentation purposes."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# ============================================================================
# Models
# ============================================================================

class SemanticSearchRequest(BaseModel):
    query_text: str = "AI là gì?"
    limit: int = 1


class SemanticSearchResult(BaseModel):
    publication_id: int
    title: str
    similarity_score: float
    chunk_preview: str


class RecommendationRequest(BaseModel):
    user_id: int = 1
    limit: int = 5


class PublicationRecommendation(BaseModel):
    publication_id: int
    title: str
    strategy: str  # "ALS" or "COLD_START"
    score: float


class HealthResponse(BaseModel):
    status: str
    database: str
    vector_store: str


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LMS AI Gateway",
    description="""
    LMS AI Gateway cung cấp các dịch vụ AI cho hệ thống thư viện:
    
    1. **Tìm kiếm ngữ nghĩa**: Tìm sách dựa trên ý nghĩa của câu hỏi (không chỉ từ khóa)
    2. **Gợi ý sách**: Gợi ý sách phù hợp cho người dùng dựa trên lịch sử tương tác
    
    Sử dụng công nghệ: SBERT embeddings + pgvector + ALS recommendations
    """,
    version="1.0.0",
)


@app.get("/health", tags=["Health"])
async def health_check() -> HealthResponse:
    """Kiểm tra trạng thái server và các dependencies."""
    return HealthResponse(
        status="ok",
        database="connected",
        vector_store="ready"
    )


@app.post(
    "/api/v1/semantic-search",
    response_model=List[SemanticSearchResult],
    tags=["Search"],
    summary="Tìm kiếm ngữ nghĩa",
    description="""
    Tìm kiếm sách dựa trên ý nghĩa của câu hỏi (semantic search).
    
    Quy trình:
    1. Mã hóa query thành vector 768 chiều bằng SBERT
    2. Tính cosine similarity với tất cả vectors trong database
    3. Trả về top-k sách có similarity cao nhất
    
    **Ví dụ request:**
    ```json
    {
        "query_text": "Machine learning là gì?",
        "limit": 5
    }
    ```
    """
)
async def semantic_search(request: SemanticSearchRequest) -> List[SemanticSearchResult]:
    """
    Thực hiện tìm kiếm ngữ nghĩa trên các sách trong thư viện.
    
    - **query_text**: Câu hỏi hoặc từ khóa tìm kiếm
    - **limit**: Số lượng kết quả trả về (mặc định: 10, tối đa: 50)
    
    **Response:** Danh sách sách được xếp hạng theo độ tương tự
    """
    return [
        SemanticSearchResult(
            publication_id=1,
            title="Introduction to Machine Learning",
            similarity_score=0.92,
            chunk_preview="Machine Learning là một nhánh của AI cho phép máy tính học từ dữ liệu..."
        ),
        SemanticSearchResult(
            publication_id=2,
            title="Deep Learning Fundamentals",
            similarity_score=0.88,
            chunk_preview="Deep Learning sử dụng mạng nơ-ron nhân tạo để xử lý dữ liệu phức tạp..."
        )
    ]


@app.post(
    "/api/v1/recommendations",
    response_model=List[PublicationRecommendation],
    tags=["Recommendations"],
    summary="Gợi ý sách",
    description="""
    Gợi ý sách phù hợp cho người dùng dựa trên:
    1. **ALS (Alternating Least Squares)**: Mô hình collaborative filtering trên user-item interactions
    2. **Cold-start Strategy**: Với user mới, gợi ý sách được xem/mượn nhiều nhất
    
    **Ví dụ request:**
    ```json
    {
        "user_id": 123,
        "limit": 10
    }
    ```
    """
)
async def get_recommendations(request: RecommendationRequest) -> List[PublicationRecommendation]:
    """
    Lấy danh sách sách được gợi ý cho người dùng.
    
    - **user_id**: ID của người dùng cần gợi ý
    - **limit**: Số lượng sách gợi ý (mặc định: 10, tối đa: 50)
    
    **Response:** Danh sách sách được gợi ý xếp theo độ phù hợp
    """
    return [
        PublicationRecommendation(
            publication_id=5,
            title="Algorithms: Design and Analysis",
            strategy="ALS",
            score=0.87
        ),
        PublicationRecommendation(
            publication_id=8,
            title="Data Structures Fundamentals",
            strategy="ALS",
            score=0.84
        ),
        PublicationRecommendation(
            publication_id=12,
            title="Software Engineering Best Practices",
            strategy="COLD_START",
            score=0.79
        )
    ]


@app.get("/api/v1/status", tags=["Health"])
async def get_status():
    """Lấy trạng thái chi tiết của AI engine."""
    return {
        "etl_status": "ready",
        "total_embeddings": 1245,
        "total_publications": 5,
        "last_etl_run": "2026-05-11T14:30:00Z",
        "vector_store": "PostgreSQL + pgvector",
        "embedding_model": "bkai-foundation-models/vietnamese-bi-encoder (768 dims)",
        "recommendation_model": "ALS (Implicit)",
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
