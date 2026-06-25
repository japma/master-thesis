from matplotlib import pyplot as plt

from dataset_loaders import build_data_loaders
from models.autoencoder import TinyAutoencoderWrapper
from utils import resolve_device
from utils.config import load_config


def main():
    ae = TinyAutoencoderWrapper()

    cfg, _ = load_config()
    dataset_cfg = cfg.dataset

    dataloader, _ = build_data_loaders(dataset_cfg)

    device = resolve_device()
    ae.to(device)

    inputs = next(iter(dataloader))[0][:16].to(device)
    print(inputs.shape)
    latent = ae.encode(inputs)
    print(latent.shape)
    recon = ae.decode(latent)
    print(recon.shape)

    plt.imshow(inputs[0].permute(1, 2, 0).cpu().detach().numpy())
    plt.show()
    plt.imshow(recon[0].permute(1, 2, 0).cpu().detach().numpy())

    plt.show()


if __name__ == "__main__":
    main()
