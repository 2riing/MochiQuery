from pymilvus import Collection, connections

# ✅ Milvus 연결
connections.connect("default", host="localhost", port="19530")

# ✅ 컬렉션별 인덱스 설정 정의
def get_index_params(collection_name):
    if collection_name == "biz_term_collection":
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 32, "efConstruction": 200}
        }
    elif collection_name == "column_meta_collection":
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
    elif collection_name == "sql_fewshot_collection":
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 32, "efConstruction": 200}
        }
    elif collection_name == "ddl_collection":
        return {
            "index_type": "IVF_PQ",
            "metric_type": "COSINE",
            "params": {"nlist": 1024, "m": 16}
        }
    elif collection_name == "table_meta_collection":
        return {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 256}
        }
    else:
        return {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 256}
        }

# ✅ 인덱스 생성 함수
def create_index_for_all():
    collections = [
        "biz_term_collection",
        "column_meta_collection",
        "sql_fewshot_collection",
        "ddl_collection",
        "table_meta_collection"
    ]

    for name in collections:
        col = Collection(name)
        index_params = get_index_params(name)
        print(f"🔧 Creating index for '{name}' with params: {index_params}")
        col.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ Index created for '{name}'")

if __name__ == "__main__":
    create_index_for_all()
    print("🎉 모든 컬렉션 인덱스 생성 완료!")

