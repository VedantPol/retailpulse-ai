from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    store_id: str = Field(..., examples=["STORE_001"])
    product_id: str = Field(..., examples=["SKU_001"])
    horizon_days: int = Field(30, ge=1, le=90)


class RecommendationRequest(BaseModel):
    product_id: str = Field(..., examples=["SKU_001"])
    top_k: int = Field(5, ge=1, le=20)


class AnomalyRequest(BaseModel):
    store_id: str = Field(..., examples=["STORE_001"])
    product_id: str = Field(..., examples=["SKU_001"])


class InsightRequest(BaseModel):
    store_id: str = Field(..., examples=["STORE_001"])
    product_id: str = Field(..., examples=["SKU_001"])
    horizon_days: int = Field(30, ge=1, le=90)

