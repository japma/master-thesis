from dataset_loaders.helpers import download_datasets


def main() -> None:
    print("Downloading datasets")
    download_datasets()
    print("Done")


if __name__ == "__main__":
    main()
