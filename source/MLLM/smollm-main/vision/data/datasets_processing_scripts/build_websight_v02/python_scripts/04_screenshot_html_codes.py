import json
import multiprocessing
import os
import re
import time
from io import BytesIO
from typing import List

from datasets import Dataset
from PIL import Image
from playwright.sync_api import sync_playwright
from tqdm import tqdm


NUM_JSON_IDEAS_AND_HTML_CODES = 20_030
NUM_PROCESSES = 60
SIZE_CHUNK_MULTIPROCESSING = 250
JPG_QUALITY = 65
COMMON_PATH_JSON_IDEAS_AND_HTML_CODES = "/home/ubuntu/hugo/websight_v02/json_ideas_and_html_codes/{IDX}.json"
COMMON_PATH_SAVE_DATASETS_PAIRS_IMAGE_CODE = "/home/ubuntu/hugo/websight_v02/datasets_pairs_image_code/{IDX}"
USER_AGENT_PLAYWRIGHT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
)


class ScreenshotHTMLcodes:
    def __init__(
        self,
        num_json_ideas_and_html_codes: int,
        num_processes: int,
        size_chunk_multiprocessing: int,
        jpg_quality: int,
        common_path_json_ideas_and_html_codes: str,
        common_path_save_datasets_pairs_image_code: str,
        user_agent_playwright: str,
    ) -> None:
        self.num_json_ideas_and_html_codes = num_json_ideas_and_html_codes
        self.num_processes = num_processes
        self.size_chunk_multiprocessing = size_chunk_multiprocessing
        self.jpg_quality = jpg_quality
        self.common_path_json_ideas_and_html_codes = common_path_json_ideas_and_html_codes
        self.common_path_save_datasets_pairs_image_code = common_path_save_datasets_pairs_image_code
        self.user_agent_playwright = user_agent_playwright

    def __call__(self) -> None:
        remaining_indices = self._check_remaining_indices()
        self._sequential_process_multi_json_ideas_and_html_codes(remaining_indices=remaining_indices)

    def _check_remaining_indices(self) -> List[int]:
        """When taking screenshots of some websites, it often fails for some reasons.
        Therefore, we do a try/except and skip the indices of the json where it failed.
        We can go through them again to increase our success rate.
        This function checks the indices of the json files that are yet to be processed.
        """
        remaining_indices = [
            idx
            for idx in range(self.num_json_ideas_and_html_codes)
            if not os.path.exists(self.common_path_save_datasets_pairs_image_code.format(IDX=idx))
        ]
        return remaining_indices

    def _load_json_ideas_and_html_codes(self, idx_json_file: int) -> List[List[str]]:
        path_json_ideas_and_html_codes = self.common_path_json_ideas_and_html_codes.format(IDX=idx_json_file)
        with open(path_json_ideas_and_html_codes, "r") as f:
            ideas_and_html_codes = json.load(f)
        return ideas_and_html_codes

    def _modify_image_urls(self, html_code: str) -> str:
        """When an image URL appears more than once, when the HTML is rendered,
        the same image is displayed. The trick is to add a `_` at the end of the
        keyword of the URL to generate another image, still corresponding to the
        same keyword.
        """
        pattern = re.compile(r"https://source\.unsplash\.com/random/\d+x\d+/\?(\w+)")
        url_counts = {}

        def replace_url(match):
            keyword = match.group(1)
            url_counts[keyword] = url_counts.get(keyword, 0) + 1
            modified_keyword = keyword + "_" * (url_counts[keyword] - 1)
            return match.group(0).replace(keyword, modified_keyword)

        modified_html_code = pattern.sub(replace_url, html_code)
        return modified_html_code

    def _convert_png_to_jpg(self, bytes_image_png: bytes, jpg_quality: int) -> Image.Image:
        image_png = Image.open(BytesIO(bytes_image_png))
        image_png = image_png.convert("RGB")
        output = BytesIO()
        image_png.save(output, format="JPEG", quality=jpg_quality)
        bytes_image_jpg = output.getvalue()
        image_jpg = Image.open(BytesIO(bytes_image_jpg))
        return image_jpg

    def _save_dataset_pairs_image_code(
        self, ideas_and_html_codes: List[List[str]], screenshots: List[Image.Image], idx_json: int
    ) -> None:
        if len(ideas_and_html_codes) != len(screenshots):
            raise ValueError("`ideas_and_html_codes` and `screenshots` should have the same length")
        ideas = [idea for idea, _ in ideas_and_html_codes]
        html_codes = [html_code for _, html_code in ideas_and_html_codes]
        ds = Dataset.from_dict({"image": screenshots, "html_code": html_codes, "llm_generated_idea": ideas})
        path_save_datasets_pairs_image_code = self.common_path_save_datasets_pairs_image_code.format(IDX=idx_json)
        ds.save_to_disk(path_save_datasets_pairs_image_code)

    def _process_one_json_ideas_and_html_codes(self, idx_json_to_process: int) -> int:
        ideas_and_html_codes = self._load_json_ideas_and_html_codes(idx_json_file=idx_json_to_process)

        screenshots = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=self.user_agent_playwright, device_scale_factor=2)
            for _, html_code in ideas_and_html_codes:
                page = context.new_page()
                html_code_modif_image_urls = self._modify_image_urls(html_code=html_code)
                page.set_content(html_code_modif_image_urls)
                page.wait_for_load_state("networkidle")
                screenshot_data = page.screenshot(full_page=True)
                image_jpg = self._convert_png_to_jpg(bytes_image_png=screenshot_data, jpg_quality=self.jpg_quality)
                screenshots.append(image_jpg)
            context.close()
            browser.close()

        self._save_dataset_pairs_image_code(
            ideas_and_html_codes=ideas_and_html_codes, screenshots=screenshots, idx_json=idx_json_to_process
        )
        return idx_json_to_process

    def _process_multi_json_ideas_and_html_codes(self, indices_to_process: List[int]) -> None:
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            with tqdm(total=len(indices_to_process)) as pbar:
                for idx_json in pool.imap_unordered(self._process_one_json_ideas_and_html_codes, indices_to_process):
                    print(f"Index {idx_json} done")
                    pbar.update()

    def _sequential_process_multi_json_ideas_and_html_codes(self, remaining_indices: List[int]) -> None:
        for idx_chunk in tqdm(range(0, len(remaining_indices), self.size_chunk_multiprocessing)):
            time.sleep(10)
            try:
                self._process_multi_json_ideas_and_html_codes(
                    indices_to_process=remaining_indices[idx_chunk : idx_chunk + self.size_chunk_multiprocessing]
                )
            except Exception:
                pass


if __name__ == "__main__":
    screenshot_html_codes = ScreenshotHTMLcodes(
        num_json_ideas_and_html_codes=NUM_JSON_IDEAS_AND_HTML_CODES,
        num_processes=NUM_PROCESSES,
        size_chunk_multiprocessing=SIZE_CHUNK_MULTIPROCESSING,
        jpg_quality=JPG_QUALITY,
        common_path_json_ideas_and_html_codes=COMMON_PATH_JSON_IDEAS_AND_HTML_CODES,
        common_path_save_datasets_pairs_image_code=COMMON_PATH_SAVE_DATASETS_PAIRS_IMAGE_CODE,
        user_agent_playwright=USER_AGENT_PLAYWRIGHT,
    )
    screenshot_html_codes()
