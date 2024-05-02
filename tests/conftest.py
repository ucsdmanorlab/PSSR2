def pytest_collection_modifyitems(session, config, items):
    items[:] = [func for func in items if func.name != "test_metrics"] # This guy is NOT a test.
