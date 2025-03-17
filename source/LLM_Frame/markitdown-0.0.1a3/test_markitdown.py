# from markitdown import MarkItDown
#
# md = MarkItDown()
# result = md.convert("/Users/gatilin/MyWork/Source-Code-Reading1/source/LLM/markitdown-0.0.1a3/tests/test_files/MMAD- THE FIRST-EVER COMPREHENSIVE BENCHMARK FOR MULTIMODAL LARGE LANGUAGE MODELS IN INDUSTRIAL ANOMALY DETECTION.pdf")
# print(result.text_content)

from markitdown import MarkItDown
import os


class MarkdownConverter:
    def __init__(self):
        self.md = MarkItDown()

    def convert(self, file_path):
        """
        Convert a file to Markdown format.

        :param file_path: Path to the file to be converted.
        :return: Converted Markdown text content.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Convert the file using MarkItDown
        result = self.md.convert(file_path)
        return result.text_content


if __name__ == "__main__":
    # Example usage
    converter = MarkdownConverter()

    # Specify the file path
    # file_path = "/Users/gatilin/MyWork/Source-Code-Reading1/source/LLM/markitdown-0.0.1a3/tests/test_files/MMAD- THE FIRST-EVER COMPREHENSIVE BENCHMARK FOR MULTIMODAL LARGE LANGUAGE MODELS IN INDUSTRIAL ANOMALY DETECTION.pdf"
    # file_path = '/Users/gatilin/MyWork/Source-Code-Reading1/source/LLM/markitdown-0.0.1a3/tests/test_files/test.pptx'
    # file_path = '/Users/gatilin/MyWork/Source-Code-Reading1/source/LLM/markitdown-0.0.1a3/tests/test_files/绿色家常食谱258例.pdf'
    file_path = '/Users/gatilin/MyWork/Source-Code-Reading1/source/LLM/markitdown-0.0.1a3/tests/test_files/test.jpg'
    try:

        markdown_content = converter.convert(file_path)
        print(markdown_content)
    except Exception as e:
        print(f"An error occurred: {e}")