import logging
import os
import re
from typing import List, Optional

import omegaconf
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from tika import parser


class EPUBProcessor(BaseLoader):
    """Processes EPUB files by extracting and cleaning text.

    This class is responsible for reading EPUB files from a specified directory, extracting their text content,
    and applying various preprocessing steps such as URL removal, non-alphanumeric filtering, stopword removal,
    and lemmatization.

    Attributes:
        cfg (dict): Configuration dictionary containing preprocessing settings.
        logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.
    """

    def __init__(
        self, cfg: omegaconf.dictconfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """Initializes the EPUBProcessor.

        Args:
            cfg (omegaconf.dictconfig): Configuration dictionary containing preprocessing settings.
            logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def _preprocess_text(self, text: str) -> str:
        """Cleans and preprocesses the extracted text.

        This method applies a series of preprocessing steps to the raw text extracted from an EPUB file,
        including URL removal, non-alphanumeric character filtering, whitespace normalization,
        and consecutive non-alphanumeric character handling.

        Args:
            text (str): Raw text extracted from an EPUB file.

        Returns:
            str: The cleaned and preprocessed text.
        """
        text = re.sub(
            self.cfg.preprocessing.whitespace,
            self.cfg.preprocessing.whitespace_replacement,
            text,
        )
        text = re.sub(
            self.cfg.preprocessing.url, self.cfg.preprocessing.url_replacement, text
        )
        text = re.sub(
            self.cfg.preprocessing.non_alphanumeric,
            self.cfg.preprocessing.non_alphanumeric_replacement,
            text,
        )
        text = re.sub(
            self.cfg.preprocessing.consecutive_non_alphanumeric,
            self.cfg.preprocessing.consecutive_non_alphanumeric_replacement,
            text,
        )
        text = re.sub(
            self.cfg.preprocessing.normalizes_newlines,
            self.cfg.preprocessing.two_lines,
            text,
        )

        return text

    def load(self) -> List[Document]:
        """Loads and processes EPUB files from the specified directory.

        This method scans the directory for EPUB files, extracts their content using the Tika parser,
        preprocesses the text using `_preprocess_text`, and returns the cleaned text as LangChain `Document` objects.

        Args:
            None

        Returns:
            List[Document]: A list of LangChain `Document` objects containing extracted and cleaned text.

        Raises:
            FileNotFoundError: If the specified directory does not exist or contains no EPUB files.
            ValueError: If no text is extracted from any EPUB file.
        """

        if not os.path.isdir(self.cfg.preprocessing.path):
            raise FileNotFoundError(
                f"Directory not found: {self.cfg.preprocessing.path}"
            )

        epub_files = [
            os.path.join(self.cfg.preprocessing.path, file)
            for file in os.listdir(self.cfg.preprocessing.path)
            if file.endswith(self.cfg.preprocessing.file_extension)
        ]

        if not epub_files:
            raise FileNotFoundError(
                f"No epub files found in {self.cfg.preprocessing.path}"
            )

        extracted_documents = []

        for epub_file in epub_files:
            book_name = os.path.splitext(os.path.basename(epub_file))[0]
            self.logger.info(f"Processing {book_name}")

            try:
                raw_text = (
                    parser.from_file(epub_file).get("content", "").lower().strip()
                )

                if not raw_text:
                    self.logger.warning(f"No text extracted from {book_name}")
                    continue

                cleaned_text = self._preprocess_text(raw_text)
                extracted_documents.append(
                    Document(page_content=cleaned_text, metadata={"source": book_name})
                )
                self.logger.info("Successfull!\n")

            except Exception as e:
                self.logger.error(f"Error Processing {book_name}: {e}", exc_info=True)

        if not extracted_documents:
            raise ValueError(
                f"No valid text extracted from {self.cfg.preprocessing.path}"
            )

        return extracted_documents
