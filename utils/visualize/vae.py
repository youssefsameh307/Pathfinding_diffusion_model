from matplotlib import pyplot as plt

def plot_vae_reconstruciton(sample, reconstructed):
    assert reconstructed.shape == sample.shape , f'{reconstructed.shape} != {sample.shape}'
    # visualize const vs reconstructed
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    pos = ax[0].imshow(sample[0].squeeze().T.cpu().detach(), origin='lower', extent=[0, 10, 0, 10], cmap='viridis')
    ax[0].set_title('Original')
    fig.colorbar(pos, ax=ax[0])
    pos = ax[1].imshow(reconstructed[0].squeeze().T.cpu().detach().numpy(), origin='lower', extent=[0, 10, 0, 10], cmap='viridis')
    ax[1].set_title('Reconstructed')
    fig.colorbar(pos, ax=ax[1])
    return fig
