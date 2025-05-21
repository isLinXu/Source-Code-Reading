import numpy as np
from PIL import Image
from scipy.fftpack import dct
from tqdm import tqdm


class ImageDeduplicator:
    @staticmethod
    def perceptual_hashing(image_dataset, num_proc):
        def func_map_perceptual_hashing(example):
            img = example["image"]
            img = img.convert("L")  # Convert to grayscale
            img = img.resize((32, 32), Image.ANTIALIAS)  # Resize to 32x32
            img_array = np.asarray(img)  # Convert to numpy array
            dct_coef = dct(dct(img_array, axis=0), axis=1)  # Compute DCT
            dct_reduced_coef = dct_coef[:8, :8]  # Retain top-left 8x8 DCT coefficients
            # Median of DCT coefficients excluding the DC term (0th term)
            median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])
            # Mask of all coefficients greater than median of coefficients
            hash = (dct_reduced_coef >= median_coef_val).flatten() * 1
            example["hash"] = hash
            return example

        image_dataset = image_dataset.map(func_map_perceptual_hashing, remove_columns="image", num_proc=num_proc)
        return image_dataset

    @staticmethod
    def hamming_distance(array_1, array_2):
        return len([1 for el_1, el_2 in zip(array_1, array_2) if el_1 != el_2])

    @staticmethod
    def brute_force_search_to_reference(hash_image_dataset, hash_image_dataset_ref, hamming_distance_threshold):
        # Compare every hash in `hash_image_dataset` to every hash of the reference
        # dataset `hash_image_dataset_ref`, and returns the indices of the rows that are
        # duplicates of an image inside the reference dataset.

        indices_duplicated_rows = []
        for i in tqdm(range(hash_image_dataset.num_rows)):
            for j in range(hash_image_dataset_ref.num_rows):
                if (
                    ImageDeduplicator.hamming_distance(
                        hash_image_dataset[i]["hash"], hash_image_dataset_ref[j]["hash"]
                    )
                    < hamming_distance_threshold
                ):
                    indices_duplicated_rows.append(i)
                    break
        return indices_duplicated_rows
