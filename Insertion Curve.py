## Code implementation for insertion curve analysis 
import numpy as np
import matplotlib.pyplot as plt
# Insertion Curve Function
def get_insertion_curve(model, image, cam, class_idx, steps=100):
    sorted_indices = np.argsort(cam.flatten())[::-1]
    scores = []
    insertion_image = np.zeros_like(image)

    for i in range(steps + 1):
        fraction_inserted = i / steps
        num_pixels_to_insert = int(fraction_inserted * cam.size)
        pixels_to_insert = sorted_indices[:num_pixels_to_insert]

        insertion_image_flat = insertion_image.flatten()
        image_flat = image.flatten()
        insertion_image_flat[pixels_to_insert] = image_flat[pixels_to_insert]
        insertion_image = insertion_image_flat.reshape(image.shape)

        score = model.predict(insertion_image[None, ...])[0, class_idx]
        scores.append(score)

    x = np.linspace(0, 1, steps + 1)
    auci = np.trapz(scores, x)

    plt.plot(np.linspace(0, 1, steps + 1), scores)
    plt.xlabel('Fraction of pixels inserted')
    plt.ylabel('Prediction score')
    plt.title('Insertion Curve')
    plt.show()

    return auci
