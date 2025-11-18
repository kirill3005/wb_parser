from recsys.candidates.covis import CoVisGenerator
from recsys.dataio.adapters import load_datasets
from recsys.dataio.schema import Schema


def test_covis_deterministic():
    schema = Schema.from_yaml("recsys/configs/schema.yaml")
    data = load_datasets(
        schema=schema,
        path_interactions="recsys/tests/fixtures/tiny_interactions.csv",
        path_items="recsys/tests/fixtures/tiny_items.csv",
        path_queries="recsys/tests/fixtures/tiny_queries.csv",
    )
    gen1 = CoVisGenerator()
    gen1.fit(data.interactions, data.items, cutoff_ts=data.interactions["ts"].max(), schema=schema, rng=None)
    res1 = gen1.score(data.queries, k=5, schema=schema)

    gen2 = CoVisGenerator()
    gen2.fit(data.interactions, data.items, cutoff_ts=data.interactions["ts"].max(), schema=schema, rng=None)
    res2 = gen2.score(data.queries, k=5, schema=schema)

    assert res1.equals(res2)
