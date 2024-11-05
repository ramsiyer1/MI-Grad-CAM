## Code implementation for deletion curve analysis
import numpy as np
import matplotlib.pyplot as plt
# Deletion Curve Function
def get_deletion_curve(model, image, cam, class_idx, steps=100):
    sorted_indices = np.argsort(cam.flatten())[::-1]
    scores = []
    deletion_image = image.copy()

    for i in range(steps + 1):
        fraction_deleted = i / steps
        num_pixels_to_delete = int(fraction_deleted * cam.size)
        pixels_to_delete = sorted_indices[:num_pixels_to_delete]

        deletion_image_flat = deletion_image.flatten()
        deletion_image_flat[pixels_to_delete] = 0
        deletion_image = deletion_image_flat.reshape(image.shape)

        score = model.predict(deletion_image[None, ...])[0, class_idx]
        scores.append(score)

    x = np.linspace(0, 1, steps + 1)
    auc = np.trapz(scores, x)

    plt.plot(np.linspace(0, 1, steps + 1), scores)
    plt.xlabel('Fraction of pixels deleted')
    plt.ylabel('Prediction score')
    plt.title('Deletion Curve')
    plt.show()

    return auc
