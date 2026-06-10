from utils.config import load_config


def main():
    cfg = load_config()
    print(f"Loaded config from {cfg}")
    print(cfg)


if __name__ == "__main__":
    main()
