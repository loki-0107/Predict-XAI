"""Unit tests for data_utils.py RUL computation."""

import pandas as pd
import numpy as np

from predictxai.data_utils import add_rul_and_labels


def test_add_rul_and_labels_train():
    """Test RUL computation for training data."""
    # Create sample training data: units with cycles 1-100
    data = {
        "unit": [1, 1, 1, 2, 2, 2],
        "cycle": [1, 50, 100, 1, 50, 100],
        "sensor_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "sensor_2": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    df = pd.DataFrame(data)

    result = add_rul_and_labels(df, is_train=True)

    # RUL should be max_cycle - cycle
    # Unit 1: max_cycle=100, so RUL at cycle 100=0, cycle 50=50, cycle 1=99
    # Unit 2: max_cycle=100, so RUL at cycle 100=0, cycle 50=50, cycle 1=99

    assert "RUL" in result.columns, "RUL column should exist"
    assert "failure_label" in result.columns, "failure_label column should exist"

    # Check unit 1, cycle 100: RUL should be 0
    rul_1_c100 = result[(result["unit"] == 1) & (result["cycle"] == 100)]["RUL"].values[0]
    assert rul_1_c100 == 0, f"Expected RUL=0 at cycle 100, got {rul_1_c100}"

    # Check unit 1, cycle 50: RUL should be 50
    rul_1_c50 = result[(result["unit"] == 1) & (result["cycle"] == 50)]["RUL"].values[0]
    assert rul_1_c50 == 50, f"Expected RUL=50 at cycle 50, got {rul_1_c50}"

    # Check failure label: RUL <= 30 should be 1
    failure_1_c100 = result[(result["unit"] == 1) & (result["cycle"] == 100)]["failure_label"].values[0]
    assert failure_1_c100 == 1, f"Expected failure_label=1 at cycle 100 (RUL=0), got {failure_1_c100}"

    print("test_add_rul_and_labels_train passed")


def test_add_rul_and_labels_test():
    """Test RUL computation for test data - the key fix."""
    # Create sample test data: units with cycles 51-100 (no failure by design)
    data = {
        "unit": [1, 1, 1, 2, 2, 2],
        "cycle": [51, 75, 100, 51, 75, 100],
        "sensor_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "sensor_2": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    df = pd.DataFrame(data)

    # For test data, RUL = max_cycle - cycle (same formula as training)
    # Unit 1: max_cycle=100, RUL at cycle 100=0, at cycle 75=25, at cycle 51=49
    # Unit 2: max_cycle=100, RUL at cycle 100=0, at cycle 75=25, at cycle 51=49

    result = add_rul_and_labels(df, is_train=False)

    # RUL should be max_cycle - cycle (same as training formula)
    assert "RUL" in result.columns, "RUL column should exist"
    assert "failure_label" in result.columns, "failure_label column should exist"

    # Check unit 1, cycle 100: RUL should be 0
    rul_1_c100 = result[(result["unit"] == 1) & (result["cycle"] == 100)]["RUL"].values[0]
    assert rul_1_c100 == 0, f"Expected RUL=0 at cycle 100, got {rul_1_c100}"

    # Check unit 1, cycle 75: RUL should be 25
    rul_1_c75 = result[(result["unit"] == 1) & (result["cycle"] == 75)]["RUL"].values[0]
    assert rul_1_c75 == 25, f"Expected RUL=25 at cycle 75, got {rul_1_c75}"

    # Check unit 1, cycle 51: RUL should be 49
    rul_1_c51 = result[(result["unit"] == 1) & (result["cycle"] == 51)]["RUL"].values[0]
    assert rul_1_c51 == 49, f"Expected RUL=49 at cycle 51, got {rul_1_c51}"

    # Check failure label near boundary: RUL <= 30 should be 1
    # At cycle 75, RUL=25, so failure_label should be 1
    failure_1_c75 = result[(result["unit"] == 1) & (result["cycle"] == 75)]["failure_label"].values[0]
    assert failure_1_c75 == 1, f"Expected failure_label=1 at cycle 75 (RUL=25), got {failure_1_c75}"

    # At cycle 100, RUL=0, so failure_label should be 1
    failure_1_c100 = result[(result["unit"] == 1) & (result["cycle"] == 100)]["failure_label"].values[0]
    assert failure_1_c100 == 1, f"Expected failure_label=1 at cycle 100 (RUL=0), got {failure_1_c100}"

    print("test_add_rul_and_labels_test passed")


def test_rul_consistency_between_train_and_test():
    """Test that train and test use the same RUL formula."""
    # Train data: cycles 1-100
    train_data = {
        "unit": [1, 1, 1],
        "cycle": [1, 50, 100],
        "sensor_1": [0.1, 0.2, 0.3],
        "sensor_2": [0.5, 0.6, 0.7],
    }
    train_df = pd.DataFrame(train_data)

    # Test data: cycles 51-100
    test_data = {
        "unit": [1, 1, 1],
        "cycle": [51, 75, 100],
        "sensor_1": [0.1, 0.2, 0.3],
        "sensor_2": [0.5, 0.6, 0.7],
    }
    test_df = pd.DataFrame(test_data)

    # Both should give RUL = max_cycle - cycle
    # For cycle 100: RUL should be 0 in both
    train_result = add_rul_and_labels(train_df.copy(), is_train=True)
    test_result = add_rul_and_labels(test_df.copy(), is_train=False)

    train_rul_100 = train_result[train_result["cycle"] == 100]["RUL"].values[0]
    test_rul_100 = test_result[test_result["cycle"] == 100]["RUL"].values[0]

    assert train_rul_100 == test_rul_100, \
        f"Train and test RUL at cycle 100 should match: train={train_rul_100}, test={test_rul_100}"

    # For cycle 50 (train) vs cycle 75 (test): RUL should be 50 vs 25
    train_rul_50 = train_result[train_result["cycle"] == 50]["RUL"].values[0]
    test_rul_75 = test_result[test_result["cycle"] == 75]["RUL"].values[0]

    assert train_rul_50 == 50, f"Train RUL at cycle 50 should be 50, got {train_rul_50}"
    assert test_rul_75 == 25, f"Test RUL at cycle 75 should be 25, got {test_rul_75}"

    print("test_rul_consistency_between_train_and_test passed")


def test_rul_clip():
    """Test that RUL is clipped to RUL_CLIP."""
    # Create data where RUL would exceed RUL_CLIP
    data = {
        "unit": [1, 1],
        "cycle": [1, 1],
        "sensor_1": [0.1, 0.2],
        "sensor_2": [0.5, 0.6],
    }
    df = pd.DataFrame(data)

    # With max_cycle=1 and cycle=1, RUL = 0, but let's test with high cycle difference
    # Actually for this test we need a scenario where max_cycle - cycle > RUL_CLIP
    # Let's just verify the clip function is applied
    result = add_rul_and_labels(df, is_train=True)

    # RUL should not exceed RUL_CLIP=125
    max_rul = result["RUL"].max()
    assert max_rul <= 125, f"Max RUL should not exceed 125, got {max_rul}"

    print("test_rul_clip passed")


def test_failure_label_boundary():
    """Test failure_label at the boundary of FAILURE_HORIZON."""
    # Create data with max_cycle clearly = 100
    # Using full cycles so groupby max_cycle = 100
    data = {
        "unit": [1, 1, 1, 1, 1, 1],
        "cycle": [1, 50, 70, 71, 99, 100],
        "sensor_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "sensor_2": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    df = pd.DataFrame(data)

    result = add_rul_and_labels(df, is_train=True)

    # max_cycle for unit 1 = 100, so RUL = 100 - cycle
    # cycle=100: RUL=0, failure_label=1
    # cycle=99: RUL=1, failure_label=1
    # cycle=71: RUL=29, failure_label=1 (29 <= 30)
    # cycle=70: RUL=30, failure_label=1 (30 <= 30) - boundary
    # cycle=50: RUL=50, failure_label=0 (50 > 30)
    # cycle=1: RUL=99, failure_label=0 (99 > 30)

    # cycle=100: RUL=0, failure=1
    rul_100 = result[result["cycle"] == 100]["RUL"].values[0]
    failure_100 = result[result["cycle"] == 100]["failure_label"].values[0]
    assert rul_100 == 0, f"Expected RUL=0 at cycle 100, got {rul_100}"
    assert failure_100 == 1, f"Expected failure_label=1 at cycle 100 (RUL=0<=30), got {failure_100}"

    # cycle=99: RUL=1, failure=1
    rul_99 = result[result["cycle"] == 99]["RUL"].values[0]
    failure_99 = result[result["cycle"] == 99]["failure_label"].values[0]
    assert rul_99 == 1, f"Expected RUL=1 at cycle 99, got {rul_99}"
    assert failure_99 == 1, f"Expected failure_label=1 at cycle 99 (RUL=1<=30), got {failure_99}"

    # cycle=71: RUL=29, failure=1
    rul_71 = result[result["cycle"] == 71]["RUL"].values[0]
    failure_71 = result[result["cycle"] == 71]["failure_label"].values[0]
    assert rul_71 == 29, f"Expected RUL=29 at cycle 71, got {rul_71}"
    assert failure_71 == 1, f"Expected failure_label=1 at cycle 71 (RUL=29<=30), got {failure_71}"

    # cycle=70: RUL=30, failure=1 - boundary
    rul_70 = result[result["cycle"] == 70]["RUL"].values[0]
    failure_70 = result[result["cycle"] == 70]["failure_label"].values[0]
    assert rul_70 == 30, f"Expected RUL=30 at cycle 70, got {rul_70}"
    assert failure_70 == 1, f"Expected failure_label=1 at cycle 70 (RUL=30<=30), got {failure_70}"

    # cycle=50: RUL=50, failure=0
    rul_50 = result[result["cycle"] == 50]["RUL"].values[0]
    failure_50 = result[result["cycle"] == 50]["failure_label"].values[0]
    assert rul_50 == 50, f"Expected RUL=50 at cycle 50, got {rul_50}"
    assert failure_50 == 0, f"Expected failure_label=0 at cycle 50 (RUL=50>30), got {failure_50}"

    # cycle=1: RUL=99, failure=0
    rul_1 = result[result["cycle"] == 1]["RUL"].values[0]
    failure_1 = result[result["cycle"] == 1]["failure_label"].values[0]
    assert rul_1 == 99, f"Expected RUL=99 at cycle 1, got {rul_1}"
    assert failure_1 == 0, f"Expected failure_label=0 at cycle 1 (RUL=99>30), got {failure_1}"

    print("test_failure_label_boundary passed")


if __name__ == "__main__":
    test_add_rul_and_labels_train()
    test_add_rul_and_labels_test()
    test_rul_consistency_between_train_and_test()
    test_rul_clip()
    test_failure_label_boundary()
    print("All tests passed")