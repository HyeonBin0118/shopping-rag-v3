"""
멀티모달 검색 모듈
===================
이미지를 GPT-4o Vision으로 분석하여 텍스트 설명 + 카테고리를 생성하고,
카테고리 필터링 + Cohere Re-ranking으로 유사 상품을 검색합니다.

개선 이력:
- v1: 이미지 → 쿼리 생성 → ChromaDB 검색
- v2: 카테고리 필터링 추가 (Slippers 등 무관 카테고리 제외)
- v3: Cohere Re-ranking 추가 (검색 순서 정확도 향상)
"""

import base64
import json
from openai import OpenAI
import cohere


# 카테고리별 허용 검색 소스 매핑
CATEGORY_MAP = {
    "Sneakers": ["Sneakers", "Shoes"],
    "Shoes":    ["Shoes", "Sneakers"],
    "Boots":    ["Boots", "Shoes"],
    "Sandals":  ["Sandals", "Shoes"],
    "Slippers": ["Slippers", "Shoes"],
    "Jackets":  ["Jackets", "Vests"],
    "Shirts":   ["Shirts", "Baselayers"],
    "Pants":    ["Pants", "Shorts"],
    "Bags":     ["Bags"],
}


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def image_to_query_and_category(image_bytes: bytes, api_key: str) -> tuple:
    """
    GPT-4o Vision으로 이미지 분석:
    - 검색 쿼리 (영어)
    - 상품 카테고리 반환
    """
    client = OpenAI(api_key=api_key)
    base64_image = encode_image(image_bytes)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze this product image and return a JSON with two fields:
1. "query": a concise search query in English (10-20 words) describing the product
2. "category": the most specific category from this list:
   [Sneakers, Shoes, Boots, Sandals, Slippers, Jackets, Shirts, Pants, Bags, Shorts, Vests, Headwear, Socks, Other]

Focus on:
- Product type and category
- Key features (waterproof, lightweight, etc.)
- Color and material if visible
- Gender target if apparent

Respond ONLY with valid JSON, no markdown.
Example: {"query": "men's white leather low-top sneakers casual", "category": "Sneakers"}
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=100
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    return result.get("query", ""), result.get("category", "Other")


def search_with_category_filter(query: str, category: str, vectorstore, top_k: int = 20):
    """
    카테고리 필터링 기반 검색
    - GPT-4o가 판단한 카테고리와 관련된 카테고리만 검색
    - 무관한 카테고리(Slippers 등) 자동 제외
    """
    all_docs = vectorstore.similarity_search(query, k=50)

    # 카테고리 필터링
    allowed_categories = CATEGORY_MAP.get(category, [category, "Shoes"])
    filtered = [
        d for d in all_docs
        if d.metadata.get("source") == "product"
        and d.metadata.get("category") in allowed_categories
    ]

    # 필터링 결과 없으면 전체 product 소스로 fallback
    if not filtered:
        filtered = [d for d in all_docs if d.metadata.get("source") == "product"]

    return filtered[:top_k]


def rerank_docs(query: str, docs: list, cohere_api_key: str, top_n: int = 5) -> list:
    """
    Cohere Re-ranking으로 검색 결과 재정렬
    - 벡터 유사도만으로는 순서 부정확 → Cross-Encoder 방식으로 재정렬
    """
    if len(docs) <= 1:
        return docs[:top_n]

    try:
        co = cohere.ClientV2(cohere_api_key)
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[d.page_content for d in docs],
            top_n=top_n,
        )
        return [docs[r.index] for r in response.results]
    except Exception:
        return docs[:top_n]


def multimodal_product_search(image_bytes: bytes, vectorstore, openai_api_key: str, cohere_api_key: str):
    """
    이미지 기반 유사 상품 검색 메인 함수
    1. GPT-4o Vision → 쿼리 + 카테고리 생성
    2. 카테고리 필터링 검색
    3. Cohere Re-ranking

    Returns: (query, category, docs) 튜플
    """
    # 1. 이미지 분석
    query, category = image_to_query_and_category(image_bytes, openai_api_key)

    # 2. 카테고리 필터링 검색
    docs = search_with_category_filter(query, category, vectorstore)

    # 3. Re-ranking
    docs = rerank_docs(query, docs, cohere_api_key)

    return query, category, docs
