from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections)


def get_collection():
    connections.connect("default", host="localhost", port="19530")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=128
        ),
        FieldSchema(
            name="label",
            dtype=DataType.INT64
        )
    ]

    schema = CollectionSchema(fields, description="MNIST vectors")
    collection = Collection("mnist_vectors", schema)

    if not collection.has_index():
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
        )

    collection.load()
    return collection
