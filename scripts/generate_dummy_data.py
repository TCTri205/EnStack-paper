import pandas as pd
import numpy as np
import os
from pathlib import Path


def generate_dummy_data():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate random dummy data
    # Columns: func (code snippet), target (vulnerability label 0-4)

    # Train data: 10 samples
    train_df = pd.DataFrame(
        {
            "func": [f"def func_{i}():\n    return {i}" for i in range(10)],
            "target": np.random.randint(0, 5, 10),
        }
    )

    # Val data: 5 samples
    val_df = pd.DataFrame(
        {
            "func": [
                f"def val_func_{i}():\n    x = {i}\n    return x" for i in range(5)
            ],
            "target": np.random.randint(0, 5, 5),
        }
    )

    # Test data: 5 samples
    test_df = pd.DataFrame(
        {
            "func": [
                f"def test_func_{i}():\n    if {i} > 0:\n        return True"
                for i in range(5)
            ],
            "target": np.random.randint(0, 5, 5),
        }
    )

    # Save to CSV
    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    print(f"Generated dummy data in {data_dir.absolute()}")
    print("train.csv:", len(train_df))
    print("val.csv:", len(val_df))
    print("test.csv:", len(test_df))


if __name__ == "__main__":
    generate_dummy_data()
