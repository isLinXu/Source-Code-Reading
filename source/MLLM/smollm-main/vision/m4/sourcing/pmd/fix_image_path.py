from typing import Callable, Dict, List

import datasets


"""
Images can be stored in `datasets` using bytes or path to an actual file. If the a path is given, one needs to make the
path work with local setup. The way to do so we remove a prefix and replace with a environment dependent one.
`home/thomas_wang_hugginface_co/.cache/m4/...` -> f"{get_m4_cache_dir()}/..."
"""


def get_image_paths_fixer(image_column_name: str, image_path_fixer: Callable[[str], str]):
    image_feature = datasets.Image(decode=True)

    def image_paths_fixer(batch: Dict[str, List]) -> Dict[str, List]:
        # Image(decode=False) which allows the images to be `{'path': str, 'bytes': str}`
        image_dicts = batch[image_column_name]

        for image_dict in image_dicts:
            # We ignore Images that store bytes directly
            if image_dict["bytes"] is not None:
                continue

            path = image_dict["path"]
            assert path is not None
            new_path = image_path_fixer(path)
            assert new_path is not None
            # Careful that's an in-place operation, which updates the dict stored in `batch`
            image_dict["path"] = new_path

        batch[image_column_name] = [image_feature.decode_example(image_dict) for image_dict in image_dicts]
        return batch

    return image_paths_fixer
