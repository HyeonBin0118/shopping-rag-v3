"""
합성 상품 데이터 생성 스크립트
================================
이미지 검색 테스트 중 Sneakers 카테고리 데이터가 전무한 것을 발견.
GPT-4o-mini로 합성 데이터를 생성하여 검색 품질을 개선합니다.

생성 카테고리:
- Sneakers: 30개 (캐주얼, 러닝화, 흰색 스니커즈 등)
- Sandals: 10개 (여름용)
"""

import os
import json
import time
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_products(category: str, count: int, examples: str) -> list:
    """GPT-4o-mini로 상품 데이터 생성"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a product data generator for an outdoor/lifestyle clothing store.
Generate realistic product data in JSON format.
Each product must have these fields:
- name: product name (brand + model style)
- description: 2-3 sentence product description focusing on materials, features, use case
- category: product category

Respond ONLY with a JSON array, no markdown, no explanation."""
            },
            {
                "role": "user",
                "content": f"""Generate {count} realistic {category} products.
Make them diverse in terms of style, color, and use case.
Examples of style to include: {examples}

Return as JSON array like:
[{{"name": "...", "description": "...", "category": "{category}"}}]"""
            }
        ],
        max_tokens=3000,
        temperature=0.8
    )

    raw = response.choices[0].message.content.strip()
    # JSON 파싱
    raw = raw.replace("```json", "").replace("```", "").strip()
    products = json.loads(raw)
    return products


def main():
    print("=" * 50)
    print("합성 상품 데이터 생성")
    print("목적: 이미지 검색 품질 향상을 위한 Sneakers/Sandals 데이터 보강")
    print("모델: GPT-4o-mini")
    print("=" * 50)

    all_products = []

    # 1. Sneakers 생성
    print("\n[1/2] Sneakers 30개 생성 중...")
    sneaker_examples = """
    - white leather low-top casual sneakers
    - running shoes lightweight mesh
    - classic canvas high-top sneakers
    - minimalist everyday sneakers
    - athletic training shoes
    - retro style colorful sneakers
    """
    sneakers = generate_products("Sneakers", 30, sneaker_examples)
    all_products.extend(sneakers)
    print(f"  ✅ Sneakers {len(sneakers)}개 생성 완료")
    time.sleep(1)

    # 2. Sandals 생성
    print("\n[2/2] Sandals 10개 생성 중...")
    sandal_examples = """
    - sport sandals with ankle strap
    - casual flip flops
    - waterproof hiking sandals
    - leather dress sandals
    """
    sandals = generate_products("Sandals", 10, sandal_examples)
    all_products.extend(sandals)
    print(f"  ✅ Sandals {len(sandals)}개 생성 완료")

    # JSONL 형식으로 저장 (기존 chunks_translated.jsonl에 추가할 형식)
    output = []
    for i, product in enumerate(all_products):
        chunk = {
            "id": f"synthetic_{i:04d}",
            "text": f"상품명: {product['name']} 카테고리: {product['category']} 설명: {product['description']}",
            "metadata": {
                "source": "product",
                "category": product["category"],
                "synthetic": True  # 합성 데이터 표시
            }
        }
        output.append(chunk)

    # 저장
    with open("synthetic_products.jsonl", "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 총 {len(output)}개 합성 상품 저장 완료 → synthetic_products.jsonl")
    print("\n샘플 확인:")
    for item in output[:3]:
        print(f"  - {item['text'][:80]}...")


if __name__ == "__main__":
    main()
