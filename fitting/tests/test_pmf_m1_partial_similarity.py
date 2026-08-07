from tools.run_pmf_m1_partial_similarity import make_data, model_points


def test_m1_data_counts_and_measure():
    data = make_data()
    assert data["D1"].shape == (12288, 2)
    assert data["D2"].shape == (9216, 2)
    assert data["D4"].shape == (3072, 2)
    points, areas, labels = model_points(1.0, seed_base=42)
    assert points.shape == (12288, 2)
    assert abs(areas.sum() - 12.0) < 1e-10
    assert set(labels) == {0, 1, 2, 3}
