import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
data_directory = os.path.join(script_directory, 'data')
masks_directory = os.path.join(data_directory, 'masks')

masks_dict = {}
for filename in os.listdir(masks_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        mask = cv2.imread(os.path.join(masks_directory, filename), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 0).astype(np.uint8)
        masks_dict[filename] = mask_bin



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for filename in os.listdir(data_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

        # Load images
        img_rgb = cv2.imread(os.path.join(data_directory, filename))
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(masks_directory, filename), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 0).astype(np.uint8)

        # CLAHE on RGB 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_rgb_clahe = cv2.merge([
            clahe.apply(img_rgb[:, :, 0]),
            clahe.apply(img_rgb[:, :, 1]),
            clahe.apply(img_rgb[:, :, 2])
        ])

        # LAB conversion 
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(img_lab)
        l_clahe = clahe.apply(l)
        img_lab_clahe = cv2.merge([l_clahe, a, b])

        # Convert back to RGB for visualization
        img_lab_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        img_lab_clahe_rgb = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)

        # Gaussian Filter on CLAHE LAB 
        img_lab_clahe_gauss = cv2.GaussianBlur(img_lab_clahe, (5, 5), 0)
        img_lab_clahe_gauss_rgb = cv2.cvtColor(img_lab_clahe_gauss, cv2.COLOR_LAB2RGB)

        # Create mask overlay 
        mask_colored = np.zeros_like(img_rgb)
        mask_colored[:, :, 0] = mask_bin * 255  # red mask

        img_rgb_overlay = cv2.addWeighted(img_rgb, 0.7, mask_colored, 0.3, 0)
        img_rgb_clahe_overlay = cv2.addWeighted(img_rgb_clahe, 0.7, mask_colored, 0.3, 0)

        img_lab_overlay = cv2.addWeighted(img_lab_rgb, 0.7, mask_colored, 0.3, 0)
        img_lab_clahe_overlay = cv2.addWeighted(img_lab_clahe_rgb, 0.7, mask_colored, 0.3, 0)
        img_lab_clahe_gauss_overlay = cv2.addWeighted(img_lab_clahe_gauss_rgb, 0.7, mask_colored, 0.3, 0)

        # Plot 
        fig, axes = plt.subplots(3, 5, figsize=(20, 10))  # 5 columns now

        # Row 1: Originals
        axes[0, 0].imshow(img_rgb); axes[0, 0].set_title("Original RGB"); axes[0, 0].axis("off")
        axes[0, 1].imshow(img_rgb_clahe); axes[0, 1].set_title("CLAHE on RGB"); axes[0, 1].axis("off")
        axes[0, 2].imshow(img_lab_rgb); axes[0, 2].set_title("Original LAB (as RGB)"); axes[0, 2].axis("off")
        axes[0, 3].imshow(img_lab_clahe_rgb); axes[0, 3].set_title("CLAHE on L (LAB)"); axes[0, 3].axis("off")
        axes[0, 4].imshow(img_lab_clahe_gauss_rgb); axes[0, 4].set_title("Gaussian Filter (CLAHE LAB)"); axes[0, 4].axis("off")

        # Row 2: Masks
        for j in range(5):
            axes[1, j].imshow(mask, cmap="gray")
            axes[1, j].set_title("Mask"); axes[1, j].axis("off")

        # Row 3: Overlays
        axes[2, 0].imshow(img_rgb_overlay); axes[2, 0].set_title("Original + Mask"); axes[2, 0].axis("off")
        axes[2, 1].imshow(img_rgb_clahe_overlay); axes[2, 1].set_title("CLAHE RGB + Mask"); axes[2, 1].axis("off")
        axes[2, 2].imshow(img_lab_overlay); axes[2, 2].set_title("LAB + Mask"); axes[2, 2].axis("off")
        axes[2, 3].imshow(img_lab_clahe_overlay); axes[2, 3].set_title("CLAHE LAB + Mask"); axes[2, 3].axis("off")
        axes[2, 4].imshow(img_lab_clahe_gauss_overlay); axes[2, 4].set_title("Gaussian CLAHE LAB + Mask"); axes[2, 4].axis("off")

        plt.tight_layout()
        plt.show()