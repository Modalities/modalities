from pathlib import Path

from llm_gym.__main__ import Main


def test_main_init(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")
    dummy_data_path = Path(__file__).parent.parent / Path("data", "lorem_ipsum.txt")
    main = Main(dataset_path=dummy_data_path, num_epochs=1)
    main.run()


if __name__ == "__main__":
    import os

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9948"
    dummy_data_path = Path(__file__).parent.parent / Path("data", "lorem_ipsum.txt")
    main = Main(dataset_path=dummy_data_path, num_epochs=1)
    main.run()
