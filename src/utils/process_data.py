import logging
import os

from tika import parser


class ProcessData:
    def __init__(self, cfg: dict, logger: logging.Logger = None) -> None:
        self.cfg = cfg
        self.logger = logger

    def read_data(self):
        books_dir = os.path.join(os.getcwd(), self.cfg["preprocessing"]["path"])
        if not os.path.isdir(books_dir):
            raise FileNotFoundError(f"Directory not found: {books_dir}")

        epub_files = [
            os.path.join(books_dir, book)
            for book in os.list(books_dir)
            if book.endswith(".epub")
        ]

        if not epub_files:
            raise FileNotFoundError(f"No epub files were found in {books_dir}")

        extracted_books = []

        for book in epub_files:
            book_name = os.path.splitext(os.path.basename(book))[0]

            if self.logger:
                self.logger.info(f"Read in {book_name}")

            extracted_text = parser.from_file(
                filename=book,
            )["content"]
            extracted_books.append(extracted_text.strip().lower())

        combined_text = "\n\n".join(extracted_books)

        return combined_text
