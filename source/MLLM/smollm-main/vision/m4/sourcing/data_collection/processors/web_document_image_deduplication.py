import json
import logging
from random import randrange

from datasets import Dataset, Features, Image, Sequence, Value, load_from_disk

from m4.sourcing.data_collection.processors.image_deduplicator import ImageDeduplicator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WebDocumentImageDeduplication:
    def __init__(
        self,
        path_web_document_dataset_train,
        path_web_document_dataset_valid,
        num_proc,
        path_save_file_map_image_url_to_pos,
        path_images_web_document_dataset_extraction,
        path_save_dir_images_web_document_dataset_train,
        path_save_file_to_be_deduplicated,
        path_save_dir_images_evaluation_tasks_dataset,
        hamming_distance_threshold,
        type_dedup,
        path_save_dir_web_document_dataset_train_deduplicated,
    ):
        self.path_web_document_dataset_train = path_web_document_dataset_train
        self.path_web_document_dataset_valid = path_web_document_dataset_valid
        self.num_proc = num_proc
        self.path_save_file_map_image_url_to_pos = path_save_file_map_image_url_to_pos
        self.path_images_web_document_dataset_extraction = path_images_web_document_dataset_extraction
        self.path_save_dir_images_web_document_dataset_train = path_save_dir_images_web_document_dataset_train
        # key is the index of a doc, value is the index of the image within this doc
        self.to_be_deduplicated = {}
        self.path_save_file_to_be_deduplicated = path_save_file_to_be_deduplicated
        self.path_save_dir_images_evaluation_tasks_dataset = path_save_dir_images_evaluation_tasks_dataset
        self.hamming_distance_threshold = hamming_distance_threshold
        # `type_dedup` defines the type of deduplication performed.
        # If it is 'remove_all_doc', the whole web document is discarded as
        # long as it contains at least one image duplicate in it.
        # If it is 'remove_image', only the specific images that are duplicates
        # are removed from the web documents.
        if type_dedup not in ["remove_all_doc", "remove_image"]:
            raise ValueError("`type_dedup` should be 'remove_all_doc' or 'remove_image'.")
        self.type_dedup = type_dedup
        self.path_save_dir_web_document_dataset_train_deduplicated = (
            path_save_dir_web_document_dataset_train_deduplicated
        )

    def create_map_image_url_to_pos(self):
        # An image is represented by its url. We aim to create a map between the urls and
        # the positions (index of the web document and index of the image within this web
        # document) where they appear in the web document dataset.
        # It will be useful for 3 cases:
        # 1. For exact image deduplication, because the same image can appear several times.
        # 2. The current dataset containing all images obtained during the extraction
        # can be reduced if a filtering step was performed after. This will help us to have
        # the list of all urls of images used in the dataset after the filtering step.
        # 3. For the image deduplication, we'll form clusters of near duplicates and keep
        # one example in each cluster. We'll need to keep track of the positions of the
        # other images to be able to discard them. Same for removing images present in an
        # evaluation task.

        logger.info("Starting loading the web document dataset.")
        self.load_web_document_dataset_train()
        logger.info("Finished loading the web document dataset.")

        def get_image_url_pos(example):
            id_doc = example["id"]
            images = example["images"]
            metadata = json.loads(example["metadata"])
            # There can be a metadata without an image if the download failed for the image
            # The "str" are here to make everything in the list of the same type for `datasets`
            example["image_url_pos"] = [
                [meta["src"], str(id_doc), str(id_list)]
                for id_list, (img, meta) in enumerate(zip(images, metadata))
                if img
            ]
            return example

        logger.info("Starting creating the map between image urls and positions in the web document dataset.")

        image_url_pos = self.web_document_dataset_train.map(
            get_image_url_pos, remove_columns=self.web_document_dataset_train.column_names, num_proc=self.num_proc
        )
        image_url_pos = image_url_pos["image_url_pos"]
        image_url_pos = [sub_el for el in image_url_pos for sub_el in el]
        self.map_image_url_to_pos = {}
        for src, id_doc, id_list in image_url_pos:
            self.map_image_url_to_pos[src] = self.map_image_url_to_pos.get(src, []) + [(int(id_doc), int(id_list))]
        with open(self.path_save_file_map_image_url_to_pos, "w") as f:
            json.dump(self.map_image_url_to_pos, f)

        logger.info("Finished creating the map between image urls and positions in the web document dataset.")

    def build_images_web_document_dataset_train(self, reload_files=False):
        # We start with the image dataset built during the construction of the web
        # document dataset. We remove the images discarded during the filtering
        if reload_files:
            logger.info("Starting reloading files.")
            self.load_map_image_url_to_pos()
            logger.info("Finished reloading files.")

        logger.info("Starting loading the previous image dataset.")
        images_web_document_dataset_extraction = load_from_disk(self.path_images_web_document_dataset_extraction)
        logger.info("Finished loading the previous image dataset.")

        def func_filter_images_web_document_dataset_extraction(example):
            if example["url"] not in self.map_image_url_to_pos:
                return False
            return True

        def bytes_to_pil_image(example):
            example["image"] = {"path": None, "bytes": example["image"]}
            return example

        logger.info("Starting making the image dataset.")
        self.images_web_document_dataset_train = images_web_document_dataset_extraction.filter(
            func_filter_images_web_document_dataset_extraction, num_proc=self.num_proc
        )
        self.images_web_document_dataset_train = self.images_web_document_dataset_train.map(
            bytes_to_pil_image,
            features=Features({"url": Value("string"), "image": Image()}),
            num_proc=self.num_proc,
        )
        self.images_web_document_dataset_train.save_to_disk(self.path_save_dir_images_web_document_dataset_train)
        logger.info("Finished making and saving the image dataset.")

    def exact_image_deduplication(self, reload_files=False):
        if reload_files:
            logger.info("Starting reloading files.")
            self.load_map_image_url_to_pos()
            logger.info("Finished reloading files.")

        logger.info("Starting performing the exact deduplication.")

        for image_url in self.map_image_url_to_pos:
            pos = self.map_image_url_to_pos[image_url]
            if len(pos) > 1:
                keep_idx = randrange(len(pos))
                del pos[keep_idx]
                for id_doc, id_list in pos:
                    self.to_be_deduplicated[id_doc] = self.to_be_deduplicated.get(id_doc, []) + [id_list]

        with open(self.path_save_file_to_be_deduplicated, "w") as f:
            json.dump(self.to_be_deduplicated, f)

        logger.info("Finished performing the exact deduplication.")

    def build_images_evaluation_tasks_dataset(self):
        # We make a dataset containing all the images present in the validation / test sets
        # of the web document dataset, as well as the images present in the evaluation tasks
        logger.info("Starting building the image dataset of the evaluation tasks.")

        self.web_document_dataset_valid = load_from_disk(self.path_web_document_dataset_valid).cast_column(
            "images", Sequence(Image(decode=False))
        )

        self.images_evaluation_tasks_dataset = Dataset.from_dict(
            {"image": [img for doc in self.web_document_dataset_valid["images"] for img in doc if img]}
        ).cast_column("image", Image())

        self.images_evaluation_tasks_dataset.save_to_disk(self.path_save_dir_images_evaluation_tasks_dataset)

        logger.info("Finished building the image dataset of the evaluation tasks.")

    def overlap_train_eval_deduplication(self, reload_files=False, reload_to_be_deduplicated=False):
        if reload_files:
            logger.info("Starting reloading files.")
            self.load_map_image_url_to_pos()
            self.load_images_web_document_dataset_train()
            self.load_images_evaluation_tasks_dataset()
            logger.info("Finished reloading files.")
        if reload_to_be_deduplicated:
            # Useful if we want to add the result of a previous exact deduplication for example
            self.load_to_be_deduplicated()

        logger.info("Starting deduplicating the overlap between the train and the evaluation.")

        image_deduplicator = ImageDeduplicator()

        logger.info("Starting hashing the two dataset of images.")
        self.hash_images_web_document_dataset_train = image_deduplicator.perceptual_hashing(
            image_dataset=self.images_web_document_dataset_train, num_proc=self.num_proc
        )
        self.hash_images_evaluation_tasks_dataset = image_deduplicator.perceptual_hashing(
            image_dataset=self.images_evaluation_tasks_dataset, num_proc=self.num_proc
        )
        logger.info("Finished hashing the two dataset of images.")

        logger.info("Starting searching for duplicates.")

        indices_duplicated_rows = image_deduplicator.brute_force_search_to_reference(
            hash_image_dataset=self.hash_images_web_document_dataset_train,
            hash_image_dataset_ref=self.hash_images_evaluation_tasks_dataset,
            hamming_distance_threshold=self.hamming_distance_threshold,
        )

        for indice in indices_duplicated_rows:
            pos_to_remove = self.map_image_url_to_pos[self.hash_images_web_document_dataset_train[indice]["url"]]
            for id_doc, id_list in pos_to_remove:
                self.to_be_deduplicated[id_doc] = self.to_be_deduplicated.get(id_doc, []) + [id_list]

        logger.info("Finished searching for duplicates.")

        with open(self.path_save_file_to_be_deduplicated, "w") as f:
            json.dump(self.to_be_deduplicated, f)

        logger.info("Finished deduplicating the overlap between the train and the evaluation.")

    def remove_duplicates(self, reload_files=False):
        if reload_files:
            logger.info("Starting reloading files.")
            self.load_to_be_deduplicated()
            self.load_web_document_dataset_train()
            logger.info("Finished reloading files.")

        logger.info("Starting removing the duplicates from the web document dataset.")

        if self.type_dedup == "remove_all_doc":

            def func_filter_remove_duplicates(example):
                if example["id"] in self.to_be_deduplicated:
                    return False
                return True

            self.web_document_dataset_train_deduplicated = self.web_document_dataset_train.filter(
                func_filter_remove_duplicates, num_proc=self.num_proc
            )

        elif self.type_dedup == "remove_image":

            def func_map_remove_duplicates(example):
                if example["id"] in self.to_be_deduplicated:
                    pos_remove = set(self.to_be_deduplicated[example["id"]])
                    example["texts"] = [el for idx, el in enumerate(example["texts"]) if idx not in pos_remove]
                    example["images"] = [el for idx, el in enumerate(example["images"]) if idx not in pos_remove]
                    example["metadata"] = json.dumps(
                        [el for idx, el in enumerate(json.loads(example["metadata"])) if idx not in pos_remove]
                    )
                return example

            self.web_document_dataset_train_deduplicated = self.web_document_dataset_train.map(
                func_map_remove_duplicates, num_proc=self.num_proc
            )

        self.web_document_dataset_train_deduplicated.save_to_disk(
            self.path_save_dir_web_document_dataset_train_deduplicated
        )

        logger.info("Finished removing the duplicates from the web document dataset.")

    def load_web_document_dataset_train(self):
        self.web_document_dataset_train = load_from_disk(self.path_web_document_dataset_train)
        self.web_document_dataset_train = self.web_document_dataset_train.add_column(
            name="id", column=range(len(self.web_document_dataset_train))
        )

    def load_map_image_url_to_pos(self):
        with open(self.path_save_file_map_image_url_to_pos) as f:
            self.map_image_url_to_pos = json.load(f)

    def load_images_web_document_dataset_train(self):
        self.images_web_document_dataset_train = load_from_disk(self.path_save_dir_images_web_document_dataset_train)

    def load_to_be_deduplicated(self):
        with open(self.path_save_file_to_be_deduplicated) as f:
            self.to_be_deduplicated = json.load(f)
        # json saves the keys as strings even if they are ints
        self.to_be_deduplicated = {int(k): v for k, v in self.to_be_deduplicated.items()}

    def load_images_evaluation_tasks_dataset(self):
        self.images_evaluation_tasks_dataset = load_from_disk(self.path_save_dir_images_evaluation_tasks_dataset)
