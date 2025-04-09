from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union


class SyllabusItemType(Enum):
    """Enum to define document element types"""

    PARAGRAPH = "paragraph"
    TABLE = "table"


@dataclass
class TableCell:
    """
    Represents a single cell in a table

    Attributes:
        text: The text content of the cell
    """

    text: str


@dataclass
class TableRow:
    """
    Represents a row in a table

    Attributes:
        cells: List of cells in this row
    """

    cells: List[TableCell]


@dataclass
class Table:
    """
    Represents a table in the syllabus

    Attributes:
        rows: List of rows in this table
    """

    rows: List[TableRow]


@dataclass
class Paragraph:
    """
    Represents a paragraph in the syllabus

    Attributes:
        text: The text content of the paragraph
    """

    text: str


@dataclass
class SyllabusElement:
    """
    Element in a syllabus

    Attributes:
        element_type: The type of element (paragraph or table)
        content: The content of the element
    """

    element_type: SyllabusItemType
    content: Union[Paragraph, Table]

    def get_content(self) -> Union[Paragraph, Table]:
        """
        Get the content in our data model format

        Returns:
            Content as Paragraph or Table
        """
        try:
            return self.content
        except ValueError as e:
            raise ValueError(f"Unknown content type: {type(self.content)}") from e


@dataclass
class SyllabusTopic:
    """
    Represents a topic in the syllabus

    Attributes:
        title: The title of the topic
        elements: List of elements in this topic
    """

    title: str
    elements: List[SyllabusElement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert topic to dictionary representation

        Returns:
            Dictionary representation of the topic
        """
        elements_data = []
        for element in self.elements:
            content = element.get_content()

            if element.element_type is SyllabusItemType.PARAGRAPH:
                elements_data.append({"type": "paragraph", "text": content.text})
            elif element.element_type is SyllabusItemType.TABLE:
                rows_data = []
                for row in content.rows:
                    rows_data.append([cell.text for cell in row.cells])

                elements_data.append({"type": "table", "rows": rows_data})

        return {"title": self.title, "elements": elements_data}
