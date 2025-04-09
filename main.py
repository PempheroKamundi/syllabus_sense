import logging
from pathlib import Path

from document_parser.syllabus_parser import NormalSyllabusParser
from output_manager.file_output_manager import FileOutputManager
from syllabus_ai_graph import SyllabusAIGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("question_generation.log"), logging.StreamHandler()],
)

if __name__ == "__main__":
    doc_path = Path.cwd() / "chemistry_form_1_2.docx"
    save_directory = Path.cwd()
    parser = NormalSyllabusParser.from_file(file_path=doc_path)
    file_manager = FileOutputManager(directory=save_directory)
    workflow = SyllabusAIGraph(
        document_parser=parser, subject="chemistry", save_manager=file_manager
    )
    workflow.process()
