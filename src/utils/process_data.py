import logging
import os
import re
from typing import List, Optional

import nltk
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tika import parser


class EPUBProcessor(BaseLoader):
    """Processes EPUB files by extracting and cleaning text.

    This class reads EPUB files from a specified directory, extracts their text content,
    and applies preprocessing steps such as URL removal, non-alphanumeric filtering,
    stopword removal, and lemmatization.

    Attributes:
        cfg (dict): Configuration dictionary containing preprocessing settings.
        logger (logging.Logger, optional): Logger instance for logging messages.
    """

    def __init__(self, cfg: dict, logger: Optional[logging.Logger] = None) -> None:
        """Initializes the EPUBProcessor.

        Args:
            cfg (dict): Configuration dictionary containing preprocessing settings.
            logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def _preprocess_text(self, text: str) -> str:
        """Cleans and preprocesses the extracted text.

        The preprocessing steps include:
        - Removing URLs.
        - Removing non-alphanumeric characters.
        - Tokenizing words.
        - Removing stopwords.
        - Lemmatizing words.

        Args:
            text (str): Raw text extracted from an EPUB file.

        Returns:
            str: Cleaned and preprocessed text.
        """
        text = re.sub(r"www.\S+", "", text)
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        tokenize_words = word_tokenize(text=text)

        processed_words = [
            lemmatizer.lemmatize(word)
            for word in tokenize_words
            if word not in stop_words
        ]

        return " ".join(processed_words)

    def load(self) -> List[Document]:
        """Loads and processes EPUB files from the specified directory.

        Downloads necessary NLTK models, scans the directory for EPUB files, extracts their content,
        preprocesses the text, and returns them as LangChain `Document` objects.

        Returns:
            List[Document]: A list of LangChain Document objects containing extracted and cleaned text.

        Raises:
            FileNotFoundError: If the directory does not exist or contains no EPUB files.
            ValueError: If no text is extracted from any EPUB file.
        """
        for model in self.cfg["preprocessing"]["nltk"]:
            nltk.download(model)

        if not os.path.isdir(self.cfg["preprocessing"]["path"]):
            raise FileNotFoundError(
                f"Directory not found: {self.cfg['preprocessing']['path']}"
            )

        epub_files = [
            os.path.join(self.cfg["preprocessing"]["path"], file)
            for file in os.listdir(self.cfg["preprocessing"]["path"])
            if file.endswith(self.cfg["preprocessing"]["file_extension"])
        ]

        if not epub_files:
            raise FileNotFoundError(
                f"No epub files found in {self.cfg['preprocessing']['path']}"
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
                f"No valid text extracted from {self.cfg['preprocessing']['path']}"
            )

        return extracted_documents
