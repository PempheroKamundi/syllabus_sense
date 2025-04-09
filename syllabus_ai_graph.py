import json
import logging
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from _base_syllabus_ai_graph_template import (BaseSyllabusSenseGraphTemplate,
                                              State)
from data_types import (BatchSelectionNodeResponse, PlanningNodeResponse,
                        QuestionPlan, QuestionsResponse,
                        SubtopicExtractionNodeResponse, SubtopicsResponse)
from document_parser.syllabus_parser import BaseSyllabusParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("question_generation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class SyllabusAIGraph(BaseSyllabusSenseGraphTemplate):
    """Implementation of the syllabus AI graph template"""

    def __init__(
        self,
        document_parser: BaseSyllabusParser,
        subject: str,
    ):
        super().__init__(document_parser=document_parser)
        self._subject = subject

    # Node for parsing the topic and extracting subtopics
    def subtopic_extraction_node(self, state: State) -> SubtopicExtractionNodeResponse:
        """Extract subtopics and learning objectives from the topic."""
        topic_data = state.topic

        # Create a parser for subtopics
        parser = PydanticOutputParser(pydantic_object=SubtopicsResponse)

        # Create a prompt for extracting subtopics with formatting instructions
        prompt_template = """
        You are an educational content analyzer. I'm going to provide you with {subject} syllabus content, 
        and I need you to extract subtopics along with their learning objectives and other metadata.

        Here's the syllabus content for the topic:
        {topic_json}

        {format_instructions}

        Analyze this content and identify distinct subtopics as specified in the format above.
        For the topic title, get it from theme/topic table, dont use the supplied one
        """

        prompt = PromptTemplate(
            input_variables=["topic_json", "subject"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template=prompt_template,
        )

        # Format the prompt with our current topic
        formatted_prompt = prompt.format(
            topic_json=json.dumps(topic_data, indent=2), subject=self._subject
        )

        # Use the LLM to analyze the topic and extract subtopics
        message = HumanMessage(content=formatted_prompt)
        response = llm.invoke([message])

        # Try to parse the response using the Pydantic parser
        try:
            # Ensure we're parsing a string
            parsed_output = parser.parse(str(response.content))

            # Add topic title to each subtopic
            topic_title = topic_data.get("title", "Unknown Topic")

            logger.info(
                f"Extracted {len(parsed_output.subtopics)} subtopics from core element: '{topic_title}'"
            )

            # Return the extracted subtopics
            return {"subtopics": parsed_output.subtopics}

        except (ValidationError, json.JSONDecodeError, OutputParserException) as e:
            # Handle parsing errors
            logger.error(f"Error parsing subtopics: {str(e)}")
            return {"subtopics": []}

    # New node for planning questions
    def question_planning_node(self, state: State) -> PlanningNodeResponse:
        """Create a plan for generating questions across all subtopics."""
        if not state.subtopics:
            logger.warning("No subtopics available for planning questions")
            # returns an empty question plan
            return {"question_plan": QuestionPlan()}

        # Create a parser for the question plan
        parser = PydanticOutputParser(pydantic_object=QuestionPlan)

        # Create a prompt for planning questions
        prompt_template = """
        You are an educational assessment planner. I'm going to provide you with a set of Chemistry subtopics,
        and I need you to create a systematic plan for generating questions that cover these subtopics.

        Here are the subtopics to cover:
        {subtopics_json}

        {format_instructions}

        For each subtopic, create planned questions with the following considerations:
        1. Balance easy, medium, and hard difficulty levels
        2. Ensure coverage of all key concepts and learning objectives
        3. Include at least 9 questions for each subtopic, with the option to add more if needed for comprehensive coverage.
        4. Assign unique IDs to each planned question
        5. Include a brief concept_area field describing what specific concept the question will test

        Create a comprehensive plan that ensures the full curriculum is properly assessed.
        """

        prompt = PromptTemplate(
            input_variables=["subtopics_json"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template=prompt_template,
        )

        # Format the prompt with our subtopics
        subtopics_data = [subtopic.model_dump() for subtopic in state.subtopics]
        formatted_prompt = prompt.format(
            subtopics_json=json.dumps(subtopics_data, indent=2)
        )

        # Use the LLM to create a question plan
        message = HumanMessage(content=formatted_prompt)
        response = llm.invoke([message])

        # Try to parse the response using the Pydantic parser
        try:
            # Ensure we're parsing a string
            parsed_output = parser.parse(str(response.content))

            logger.info(
                f"Created question plan with {len(parsed_output.planned_questions)} total questions"
            )

            # Return the question plan
            return {"question_plan": parsed_output}

        except (ValidationError, json.JSONDecodeError, OutputParserException) as e:
            # Handle parsing errors
            logger.error(f"Error creating question plan: {str(e)}")
            return {"question_plan": QuestionPlan()}

    # Node for selecting the next batch of questions to generate
    def batch_selection_node(self, state: State) -> BatchSelectionNodeResponse:
        """Select the next batch of questions to generate from the plan."""
        if not state.question_plan or not state.question_plan.planned_questions:
            logger.warning("No question plan available for batch selection")
            return {"current_batch": []}

        # Get the starting position for this batch
        start_pos = state.plan_position

        # Check if we've reached the end of the plan
        if start_pos >= len(state.question_plan.planned_questions):
            logger.info("Reached the end of the question plan")
            return {"current_batch": []}

        # Determine the end position for this batch
        end_pos = min(
            start_pos + state.batch_size, len(state.question_plan.planned_questions)
        )

        # Get the current batch of questions
        current_batch = state.question_plan.planned_questions[start_pos:end_pos]

        # Update their status
        for question in current_batch:
            question.status = "generating"

        logger.info(
            f"Selected batch of {len(current_batch)} questions (positions {start_pos + 1} to {end_pos} of {len(state.question_plan.planned_questions)})"
        )

        # Return the batch and update position
        return {"current_batch": current_batch, "plan_position": end_pos}

    # Node for generating questions based on the current batch
    def batch_question_generation_node(self, state: State) -> Dict[str, Any]:
        """Generate questions for the current batch from the plan."""
        if not state.current_batch:
            logger.warning("No current batch available for question generation")
            return {"current_questions": []}

        # Find the subtopic for this batch (assume all questions in batch are from same subtopic)
        subtopic_name = state.current_batch[0].subtopic
        current_subtopic = None

        for subtopic in state.subtopics:
            if subtopic.subtopic_name == subtopic_name:
                current_subtopic = subtopic
                break

        if not current_subtopic:
            logger.error(f"Could not find subtopic '{subtopic_name}' for current batch")
            return {"current_questions": []}

        # Create a parser for questions
        parser = PydanticOutputParser(pydantic_object=QuestionsResponse)

        # Create a prompt for generating questions based on the plan
        prompt_template = """
        Generate multiple-choice Chemistry questions for Form 1 students based on the following planned questions:

        Subtopic: "{subtopic_name}" within the main topic "{topic_title}"

        Here's information about this subtopic:
        Learning objectives: {learning_objectives}
        Key concepts: {key_concepts}
        Assessment criteria: {assessment_criteria}

        Now, generate questions according to this specific plan:
        {planned_questions_json}

        {format_instructions}

        For each question:
        1. Include four answer choices (one correct, three incorrect)
        2. Provide a detailed explanation for the correct answer
        3. Include a helpful hint
        4. Match the difficulty level exactly as specified in the plan
        5. Address the specific concept area indicated in the plan

        Make sure each question clearly tests the concept area indicated in the plan.
        Use the exact same question_id as provided in the plan.

        Generate exactly {batch_size} questions matching the specifications in the plan.
        """

        prompt = PromptTemplate(
            input_variables=[
                "subtopic_name",
                "topic_title",
                "learning_objectives",
                "key_concepts",
                "assessment_criteria",
                "planned_questions_json",
                "batch_size",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template=prompt_template,
        )

        # Format the prompt with our current batch
        formatted_prompt = prompt.format(
            subtopic_name=current_subtopic.subtopic_name,
            topic_title=current_subtopic.topic_title,
            learning_objectives=current_subtopic.learning_objectives,
            key_concepts=current_subtopic.key_concepts,
            assessment_criteria=current_subtopic.assessment_criteria,
            planned_questions_json=json.dumps(
                [q.model_dump() for q in state.current_batch], indent=2
            ),
            batch_size=len(state.current_batch),
        )

        # Generate questions using the LLM
        message = HumanMessage(content=formatted_prompt)
        response = llm.invoke([message])

        # Try to parse the response using the Pydantic parser
        try:
            # Parse the content
            parsed_output = parser.parse(str(response.content))

            # Ensure the questions match the planned IDs
            planned_ids = {q.question_id for q in state.current_batch}
            for question in parsed_output.questions:
                if question.question_id not in planned_ids:
                    # If the ID doesn't match, find the corresponding planned question
                    for i, planned_q in enumerate(state.current_batch):
                        if i < len(parsed_output.questions):
                            # Assign the correct ID from the plan
                            question.question_id = planned_q.question_id

            logger.info(f"Generated {len(parsed_output.questions)} questions for batch")

            # Return the generated questions
            return {"current_questions": parsed_output.questions}

        except (ValidationError, json.JSONDecodeError, OutputParserException) as e:
            # Handle parsing errors
            logger.error(f"Error parsing generated questions: {str(e)}")
            return {"current_questions": []}

    # Node for saving questions
    def question_saving_node(self, state: State) -> Dict[str, Any]:
        """Save the generated questions and prepare for the next batch."""
        # If we have no questions, just return the current state
        if not state.current_questions:
            return {"questions": state.questions}

        # Add the current questions to our accumulated questions
        all_questions = state.questions + state.current_questions

        # Save the questions to a JSON file
        output_file = Path.cwd() / "generated_questions.json"

        try:
            # Check if the file exists and read existing content
            if output_file.exists():
                with open(output_file, "r") as f:
                    try:
                        existing_questions = json.load(f)
                    except json.JSONDecodeError:
                        # File exists but is not valid JSON
                        existing_questions = []
            else:
                existing_questions = []

            # Append our new questions
            # Convert Pydantic models to dict for JSON serialization
            new_questions_dict = [q.model_dump() for q in state.current_questions]
            all_saved_questions = existing_questions + new_questions_dict

            # Write back to the file
            with open(output_file, "w") as f:
                json.dump(all_saved_questions, f, indent=2)

            # Get the current subtopic name for logging (from the first question if available)
            subtopic_name = (
                state.current_questions[0].topic
                if state.current_questions
                else "Unknown Subtopic"
            )
            logger.info(
                f"Saved {len(state.current_questions)} questions for subtopic: '{subtopic_name}'"
            )

        except Exception as e:
            logger.error(f"Error saving questions: {e}")

        # Return updated questions list
        return {"questions": all_questions}

    # Node for deciding whether to continue to next batch or end
    def batch_decision_node(self, state: State) -> str:
        """Decide whether to process another batch or end."""
        if not state.question_plan or not state.question_plan.planned_questions:
            logger.warning("No question plan available for decision")
            return "end"

        # Check if we've processed all questions in the plan
        if state.plan_position >= len(state.question_plan.planned_questions):
            logger.info("All planned questions have been generated. Workflow complete.")
            return "end"
        else:
            # Continue with the next batch
            logger.info(
                f"Proceeding to next batch. {state.plan_position}/{len(state.question_plan.planned_questions)} questions processed."
            )
            return "next_batch"
