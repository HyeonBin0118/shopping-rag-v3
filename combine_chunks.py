import json

# 기존 chunks_translated.jsonl 로드
original = []
with open('chunks_translated.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        original.append(json.loads(line.strip()))

# synthetic_products.jsonl 로드 + 형식 변환
# step2_embedding.py가 기대하는 형식: doc_id, text, source, category
synthetic_raw = []
with open('synthetic_products.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        converted = {
            "doc_id": item["id"],
            "text": item["text"],
            "source": item["metadata"]["source"],
            "category": item["metadata"]["category"]
        }
        synthetic_raw.append(converted)

# 합치기
all_chunks = original + synthetic_raw

# 저장
with open('chunks_combined.jsonl', 'w', encoding='utf-8') as f:
    for item in all_chunks:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f'합치기 완료!')
print(f'원본: {len(original)}개')
print(f'합성: {len(synthetic_raw)}개')
print(f'총합: {len(all_chunks)}개')
