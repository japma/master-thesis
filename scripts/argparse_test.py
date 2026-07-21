from utils.config import load_config


def main() -> None:
    cfg = load_config()
    print(cfg)


if __name__ == "__main__":
    main()
