from __future__ import annotations

from run_models import (
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_split_datasets,
    format_period,
    load_dataset,
)


def main() -> None:
    ensure_split_datasets()
    train_panel = load_dataset(TRAIN_DATA_PATH)
    test_panel = load_dataset(TEST_DATA_PATH)

    print("Train/test-splitt generert.")
    print(f"- Train-fil: {TRAIN_DATA_PATH}")
    print(f"- Test-fil: {TEST_DATA_PATH}")
    print(f"- Train-periode: {format_period(train_panel)} | observasjoner={len(train_panel)}")
    print(f"- Test-periode: {format_period(test_panel)} | observasjoner={len(test_panel)}")


if __name__ == "__main__":
    main()
