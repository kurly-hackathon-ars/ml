from app import deps, models, service


def test__get_recommend_model():
    assert service._get_recommendation_model() is not None


def test__get_item():
    print(service._get_item(1))
    assert service._get_item(1) is not None


def test_recommend_by_activity():
    deps.setup_sample_items()
    assert service.recommend_by_activity(0)


def test__train_model():
    deps.setup_sample_items()  # TODO: Remove
    service._train_model()
    assert False
