from abc import ABC, abstractmethod

from _base_syllabus_ai_graph_template import State


class BaseOutputManager(ABC):
    """base output manager class for all output managers"""

    @abstractmethod
    def save_output(
        self, state: State
    ) -> None:
        raise NotImplementedError("save_output method not implemented")
