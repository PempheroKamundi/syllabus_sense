import logging
from pathlib import Path

from document_parser.syllabus_parser import NormalSyllabusParser
from syllabus_ai_graph import SyllabusAIGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("question_generation.log"), logging.StreamHandler()],
)

if __name__ == "__main__":
    # Example topic data - in a real scenario, this would come from your parser
    path = Path.cwd() / "chemistry_form_1_2.docx"
    parser = NormalSyllabusParser.from_file(file_path=path)
    workflow = SyllabusAIGraph(document_parser=parser, subject="chemistry")
    workflow.process()
