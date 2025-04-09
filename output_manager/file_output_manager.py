import json
import logging
from pathlib import Path

from _base_syllabus_ai_graph_template import State
from exceptions import InvalidOutputDirectoryError
from output_manager.base_output_manager import BaseOutputManager

logger = logging.getLogger(__name__)


class FileOutputManager(BaseOutputManager):
    def __init__(self, directory: Path) -> None:
        self._ensure_valid_directory(directory)
        self._dir = directory

    @staticmethod
    def _ensure_valid_directory( directory: Path) -> None:
        """
        Ensures the provided path is a valid directory, creating it if it doesn't exist.

        Args:
            directory: Path to validate and potentially create

        Raises:
            InvalidOutputDirectoryError: If the path exists but is not a directory
        """
        if not directory.exists():
            logger.info(f"Creating output directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        elif not directory.is_dir():
            raise InvalidOutputDirectoryError(directory, "Path exists but is not a directory")

    def save_output(
        self, state: State
    ) -> None:
        """
        Appends or adds new questions to the specified file path

        Args:
           state: State object containing current_questions
        """
        document_name = state.current_questions[0].topic
        output_file = self._dir / f"{document_name}.json"

        try:
            with open(output_file, "r") as f:
                # TODO : draw back of loading questions here is memory. Bigger JSON
                # files might not work well. For our use case, it should work fine for now
                existing_questions = json.load(f)
        except FileNotFoundError:
            existing_questions = []
        except json.JSONDecodeError:
            existing_questions = []

        new_questions_dict = [q.model_dump() for q in state.current_questions]
        all_saved_questions = existing_questions + new_questions_dict

        with open(output_file, "w") as f:
            json.dump(all_saved_questions, f, indent=2)

        subtopic_name = (
            state.current_questions[0].topic
            if state.current_questions
            else "Unknown Subtopic"
        )
        logger.info(
            f"Saved {len(state.current_questions)} questions for topic: '{subtopic_name}' to {output_file}"
        )
