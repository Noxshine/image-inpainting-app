import matplotlib.pyplot as plt

def display(gt_img, mask_img, masked_img, output):
    # matplotlib display
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(gt_img)
    axes[0].set_title('ground truth')
    axes[0].axis('off')

    axes[1].imshow(mask_img)
    axes[1].set_title('mask')
    axes[1].axis('off')

    axes[2].imshow(masked_img)
    axes[2].set_title('masked image')
    axes[2].axis('off')

    axes[3].imshow(output)
    axes[3].set_title('output')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()