from src.train import main

def test_accuracy_threshold():
    acc = main(0.2, 42)
    assert acc >= 0.9, f"Expected accuracy >= 0.9, got {acc}"
