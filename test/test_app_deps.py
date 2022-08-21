from app import deps


def test_deps_setup_sample_items():
    deps.setup_sample_items()
    assert deps.get_items()
