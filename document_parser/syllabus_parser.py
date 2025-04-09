from abc import ABC
from pathlib import Path
from typing import Optional, Union

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from .data_types import SyllabusElement, SyllabusItemType, SyllabusTopic


class BaseSyllabusParser(ABC):
    """Abstract base class for Syllabus parsers."""

    def __iter__(self):
        """Make SyllabusParser an iterator"""
        raise NotImplementedError("SyllabusParser must implement __iter__")

    def __next__(self) -> SyllabusTopic:
        """Get the next topic"""
        raise NotImplementedError("SyllabusParser must implement __next__")


class NormalSyllabusParser(BaseSyllabusParser):
    """
    Parses Word documents to extract SyllabusTopic objects.
    This supports the normal structure of Malawian syllabuses
    """

    def __init__(self, document: DocxDocument, topic_identifier: str = "Core element"):
        """
        Initialize the parser

        Args:
            document: The docx Document object
            topic_identifier: Text that marks the beginning of a topic
        """
        self._document = document
        self.topic_identifier = topic_identifier
        self._document_body = document._element.body
        self._topic_count = 0
        self._element_count = 0
        self._generator = None

    def _parse_document_element(self, element) -> Optional[SyllabusElement]:
        """
        Convert document element to a SyllabusElement

        Args:
            element: A docx document element

        Returns:
            SyllabusElement if recognized, None otherwise
        """

        if isinstance(element, CT_P):
            paragraph = docx.text.paragraph.Paragraph(element, self._document)
            if paragraph.text.strip():
                return SyllabusElement(
                    element_type=SyllabusItemType.PARAGRAPH, content=paragraph
                )

        if isinstance(element, CT_Tbl):
            table = docx.table.Table(element, self._document)
            return SyllabusElement(element_type=SyllabusItemType.TABLE, content=table)

        return None

    def _is_topic_marker(self, element: SyllabusElement) -> Optional[str]:
        """
        Check if element starts a new topic

        Args:
            element: The element to check

        Returns:
            Topic title if this is a marker, None otherwise
        """
        if element.element_type is SyllabusItemType.PARAGRAPH:
            text = element.content.text.strip()

            if self.topic_identifier in text:
                # Extract the topic title
                topic_title = (
                    text.replace(self.topic_identifier, "")
                    .replace(f"**{self.topic_identifier}**", "")
                    .strip()
                )
                topic_title = topic_title.strip(" -:")  # Clean up common separators
                return topic_title

        return None

    def __iter__(self):
        """Make SyllabusParser an iterator"""
        self._generator = self._process_topics()
        return self

    def __next__(self) -> SyllabusTopic:
        """
        Get the next topic

        Returns:
            SyllabusTopic object

        Raises:
            StopIteration: When no more topics
        """
        if self._generator is None:
            self._generator = self._process_topics()

        title, elements = next(self._generator)
        return SyllabusTopic(title=title, elements=elements)

    def _process_topics(self):
        """
        Process document and yield each topic

        Yields:
            Tuple of (title, elements)
        """
        current_topic_title = None
        topic_elements = []

        for element in self._document_body:
            content_element = self._parse_document_element(element)
            if not content_element:
                continue

            self._element_count += 1

            # Check if this element starts a new topic
            topic_title = self._is_topic_marker(content_element)

            if topic_title:
                # If we already have a topic in progress, finish it
                if current_topic_title:
                    yield current_topic_title, topic_elements
                    self._topic_count += 1

                # Start a new topic
                current_topic_title = topic_title
                topic_elements = [content_element]
            elif current_topic_title:
                # Add to current topic
                topic_elements.append(content_element)

        # Handle the last topic
        if current_topic_title:
            yield current_topic_title, topic_elements
            self._topic_count += 1

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], topic_identifier: str = "Core element"
    ) -> "SyllabusParser":
        """
        Create parser from a file

        Args:
            file_path: Path to the document file
            topic_identifier: Text that marks beginning of topics

        Returns:
            SyllabusParser instance
        """
        with open(file_path, "rb") as file:
            document = docx.Document(file)
            return cls(document, topic_identifier=topic_identifier)
