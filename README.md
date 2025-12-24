# Vector Database

## Resources

- https://www.pinecone.io/learn/vector-database/
- https://www.dailydoseofds.com/a-beginner-friendly-and-comprehensive-deep-dive-on-vector-databases/
- [deeplerning ai course for usages of vector-database](https://learn.deeplearning.ai/courses/building-applications-vector-databases)

## Usage

```sh
docker run -d --name milvus -p 19530:19530 -p 19121:19121 milvusdb/milvus:v2.6.7

py train.py
uvicorn app.main:app --reload
http://127.0.0.1:8000/docs
```
