import logging
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tika import parser


class EPUBProcessor:
    """
    A class for processing EPUB files by extracting and cleaning text.

    This class reads EPUB files from a specified directory, extracts their text content,
    and applies preprocessing steps such as URL removal, non-alphanumeric filtering,
    stopword removal, and lemmatization.

    Attributes:
        cfg (dict): Configuration dictionary containing preprocessing settings.
        logger (logging.Logger, optional): Logger instance for logging messages.
        combined_text (str): The combined text extracted from all EPUB files (unused in the current implementation).
    """

    def __init__(self, cfg: dict, logger: logging.Logger = None) -> None:
        """
        Initializes the EPUBProcessor with configuration settings and an optional logger.

        Args:
            cfg (dict): Configuration dictionary containing preprocessing settings.
            logger (logging.Logger, optional): Logger instance for logging messages. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger

    def process_epub_text(self) -> str:
        """
        Extracts and cleans text from EPUB files in the specified directory.

        This method:
        - Reads EPUB files from the configured directory.
        - Extracts text using Apache Tika.
        - Removes URLs and non-alphanumeric characters.
        - Downloads required NLTK resources.
        - Tokenizes words, removes stopwords, and applies lemmatization.

        Returns:
            str: The cleaned and preprocessed text from the EPUB files.

        Raises:
            FileNotFoundError: If the EPUB directory or no EPUB files are found.
            ValueError: If no text is extracted from the EPUB files.
        """

        books_dir = os.path.join(os.getcwd(), self.cfg["preprocessing"]["path"])

        if not os.path.isdir(books_dir):
            raise FileNotFoundError(f"Directory not found: {books_dir}")

        epub_files = [
            os.path.join(books_dir, book)
            for book in os.listdir(books_dir)
            if book.endswith(".epub")
        ]

        if not epub_files:
            raise FileNotFoundError(f"No epub files were found in {books_dir}")

        extracted_books = []

        for book in epub_files:
            book_name = os.path.splitext(os.path.basename(book))[0]

            if self.logger:
                self.logger.info(f"Reading {book_name}")

            extracted_text = parser.from_file(
                filename=book,
            )["content"]

            extracted_books.append(extracted_text.strip().lower())

        if not extracted_books:
            raise ValueError("No text extracted from books.")

        raw_text = "\n\n".join(extracted_books)

        text_without_urls = re.sub(r"www.\S+", "", raw_text)
        text_alpha_numeric = re.sub(r"[^A-Za-z0-9\s]", "", text_without_urls)

        for model in self.cfg["preprocessing"]["nltk"]:
            nltk.download(model)

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text_alpha_numeric)

        processed_text = " ".join(
            lemmatizer.lemmatize(word) for word in words if word not in stop_words
        )

        return processed_text
