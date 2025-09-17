import pandas as pd
import pytest

from midst_toolkit.data_processing.utils import SynthEvalDataframeEncoding


TEST_REAL_DATAFRAME = pd.DataFrame(
    {
        "column_a": [1, 2, 3, 4, 5],
        "column_b": [1, 1.5, 2.0, 3.0, -1.0],
        "column_c": ["house", "cat", "cat", "car", "dog"],
        "column_d": [1, 1, 3, 2, 2],
        "column_e": [1.0, 3.0, 1.0, 2.0, 1.0],
    }
)

TEST_SYNTHETIC_DATAFRAME = pd.DataFrame(
    {
        "column_a": [1, 2, 3, 4, 5],
        "column_b": [1, 1.5, 2.0, 3.0, -100.0],
        "column_c": ["human", "cat", "car", "car", "dog"],
        "column_d": [1, 1, 3, 2, 2],
        "column_e": [1.0, 3.0, 1.0, 2.0, 4.0],
    }
)


def test_syntheval_dataframe_encoder() -> None:
    # Column e is numeric, leave off b
    encoder = SynthEvalDataframeEncoding(
        TEST_REAL_DATAFRAME, TEST_SYNTHETIC_DATAFRAME, ["column_c", "column_d"], ["column_a", "column_e"]
    )
    encoded_real = encoder.encode(TEST_REAL_DATAFRAME)
    encoded_synthetic = encoder.encode(TEST_SYNTHETIC_DATAFRAME)

    # Min-max scaled.
    assert encoded_real["column_a"].tolist() == pytest.approx([0, 0.25, 0.5, 0.75, 1.0], abs=1e-8)
    # Should be untouched, as it wasn't included in the columns
    assert encoded_real["column_b"].tolist() == pytest.approx([1, 1.5, 2.0, 3.0, -1.0], abs=1e-8)
    # Cat mapped {car: 0, cat: 1, dog: 2, house: 3, human: 4}
    assert encoded_real["column_c"].tolist() == pytest.approx([3, 1, 1, 0, 2], abs=1e-8)
    # Cat mapped {1: 0, 2: 1, 3: 2}
    assert encoded_real["column_d"].tolist() == pytest.approx([0, 0, 2, 1, 1], abs=1e-8)
    # Min-max scaled.
    assert encoded_real["column_e"].tolist() == pytest.approx([0 / 3.0, 2 / 3.0, 0 / 3.0, 1 / 3.0, 0 / 3.0], abs=1e-8)

    # Min-max scaled.
    assert encoded_synthetic["column_a"].tolist() == pytest.approx([0, 0.25, 0.5, 0.75, 1.0], abs=1e-8)
    # Should be untouched, as it wasn't included in the columns
    assert encoded_synthetic["column_b"].tolist() == pytest.approx([1, 1.5, 2.0, 3.0, -100.0], abs=1e-8)
    # Cat mapped {car: 0, cat: 1, dog: 2, house: 3, human: 4}
    assert encoded_synthetic["column_c"].tolist() == pytest.approx([4, 1, 0, 0, 2], abs=1e-8)
    # Cat mapped {1: 0, 2: 1, 3: 2}
    assert encoded_synthetic["column_d"].tolist() == pytest.approx([0, 0, 2, 1, 1], abs=1e-8)
    # Min-max scaled.
    assert encoded_synthetic["column_e"].tolist() == pytest.approx(
        [0 / 3.0, 2 / 3.0, 0 / 3.0, 1 / 3.0, 3 / 3.0], abs=1e-8
    )

    # Column e is categorical, all included
    encoder = SynthEvalDataframeEncoding(
        TEST_REAL_DATAFRAME, TEST_SYNTHETIC_DATAFRAME, ["column_c", "column_d", "column_e"], ["column_a", "column_b"]
    )
    encoded_real = encoder.encode(TEST_REAL_DATAFRAME)
    encoded_synthetic = encoder.encode(TEST_SYNTHETIC_DATAFRAME)

    # Min-max scaled.
    assert encoded_real["column_a"].tolist() == pytest.approx([0, 0.25, 0.5, 0.75, 1.0], abs=1e-8)
    # Should now be mixmax scaled
    assert encoded_real["column_b"].tolist() == pytest.approx(
        [101 / 103, 101.5 / 103, 102 / 103, 1.0, 99 / 103], abs=1e-8
    )
    # Cat mapped {car: 0, cat: 1, dog: 2, house: 3, human: 4}
    assert encoded_real["column_c"].tolist() == pytest.approx([3, 1, 1, 0, 2], abs=1e-8)
    # Cat mapped {1: 0, 2: 1, 3: 2}
    assert encoded_real["column_d"].tolist() == pytest.approx([0, 0, 2, 1, 1], abs=1e-8)
    # Cat mapped {1: 0, 2: 1, 3: 2, 4: 3}
    assert encoded_real["column_e"].tolist() == pytest.approx([0, 2, 0, 1, 0], abs=1e-8)

    # Min-max scaled.
    assert encoded_synthetic["column_a"].tolist() == pytest.approx([0, 0.25, 0.5, 0.75, 1.0], abs=1e-8)
    # Should now be mixmax scaled
    assert encoded_synthetic["column_b"].tolist() == pytest.approx(
        [101 / 103, 101.5 / 103, 102 / 103, 1.0, 0], abs=1e-8
    )
    # Cat mapped {car: 0, cat: 1, dog: 2, house: 3, human: 4}
    assert encoded_synthetic["column_c"].tolist() == pytest.approx([4, 1, 0, 0, 2], abs=1e-8)
    # Cat mapped {1: 0, 2: 1, 3: 2}
    assert encoded_synthetic["column_d"].tolist() == pytest.approx([0, 0, 2, 1, 1], abs=1e-8)
    # Cat mapped {1: 0, 2: 1, 3: 2, 4: 3}
    assert encoded_synthetic["column_e"].tolist() == pytest.approx([0, 2, 0, 1, 3], abs=1e-8)
