"""
syllabus_sense._base_syllabus_ai_graph_template
~~~~~~~~~~~~
Contains code for the main syllabus sense program without async support
"""
import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from data_types import (BatchSelectionNodeResponse, PlannedQuestion,
                        PlanningNodeResponse, Question, QuestionPlan, Subtopic,
                        SubtopicExtractionNodeResponse)
from document_parser.syllabus_parser import BaseSyllabusParser

logger = logging.getLogger(__name__)


class State(BaseModel):
    """
    Represents the state of the syllabus sense graph.

    Attributes:
        topic: Dictionary containing information about the main topic
        subtopics: List of subtopics extracted from the syllabus
        current_subtopic: Currently active subtopic being processed
        questions: Complete list of generated questions
        current_questions: Questions related to the current subtopic
        current_subtopic_index: Index of the current subtopic in the subtopics list
        question_plan: Plan for generating questions
        plan_position: Current position in the question plan
        batch_size: Number of questions to generate per batch
        current_batch: Current batch of planned questions to be processed
    """

    topic: Dict[str, Any]
    subtopics: List[Subtopic] = Field(default_factory=list)
    current_subtopic: Optional[Subtopic] = None
    questions: List[Question] = Field(default_factory=list)
    current_questions: List[Question] = Field(default_factory=list)
    current_subtopic_index: int = 0
    question_plan: Optional[QuestionPlan] = None
    plan_position: int = 0
    batch_size: int = 5  # Number of questions to generate per batch
    current_batch: List[PlannedQuestion] = Field(default_factory=list)


class BaseSyllabusSenseGraphTemplate:
    """
    Template pattern for the graph AI agent for syllabus sense.

    Defines the structure and flow of the AI graph, providing abstract
    methods that need to be implemented by concrete subclasses.
    """

    def __init__(self, document_parser: BaseSyllabusParser) -> None:
        """
        Initialize the BaseSyllabusSenseGraphTemplate.

        Args:
            document_parser: Parser for extracting information from syllabi
        """
        self._document_parser: BaseSyllabusParser = document_parser
        self._graph: Optional[Callable] = None
        logger.info("Initialized BaseSyllabusSenseGraphTemplate")

    def _create_ai_graph_structure(self) -> None:
        """
        Creates the AI graph structure for synchronous execution.
        """
        logger.info("Creating AI graph structure")
        workflow: StateGraph = StateGraph(State)

        # Add all nodes
        logger.info("Adding nodes to the workflow")
        workflow.add_node("subtopic_extraction", self.subtopic_extraction_node)
        workflow.add_node("question_planning", self.question_planning_node)
        workflow.add_node("batch_selection", self.batch_selection_node)
        workflow.add_node(
            "batch_question_generation", self.batch_question_generation_node
        )
        workflow.add_node("question_saving", self.question_saving_node)

        # Set the entry point and connect the nodes
        logger.info("Setting entry point and connecting nodes")
        workflow.set_entry_point("subtopic_extraction")
        workflow.add_edge("subtopic_extraction", "question_planning")
        workflow.add_edge("question_planning", "batch_selection")
        workflow.add_edge("batch_selection", "batch_question_generation")
        workflow.add_edge("batch_question_generation", "question_saving")

        # Add conditional edges for batch processing
        logger.info("Adding conditional edges for batch processing")
        workflow.add_conditional_edges(
            "question_saving",
            self.batch_decision_node,
            {"next_batch": "batch_selection", "end": END},
        )

        # Compile the graph
        logger.info("Compiling the graph")
        self._graph = workflow.compile()
        logger.info("AI graph structure created successfully")

    def process(self, topics_num: int = 1) -> None:
        """
        Process the syllabus and generate questions.

        Args:
            topics_num: Number of topics to process (default: 1)
        """
        logger.info(f"Starting processing of {topics_num} topics")

        if not self._graph:
            logger.info("Graph not initialized, creating now")
            self._create_ai_graph_structure()

        # Process the specified number of topics
        processed_count = 0
        while processed_count < topics_num:
            try:
                # Get the next document from the parser
                logger.info(f"Processing document {processed_count + 1} of {topics_num}")
                next_document = next(self._document_parser)
                logger.info(
                    f"Retrieved document with title: {next_document.title if hasattr(next_document, 'title') else 'Unknown'}")

                initial_state = State(topic=next_document.to_dict())
                logger.info(f"Created initial state with topic: {initial_state.topic.get('title', 'Unknown')}")

                # Use synchronous invoke
                logger.info("Invoking graph for processing")
                final_state = self._graph.invoke(initial_state)
                logger.info(f"Processing complete. Generated {len(final_state.questions)} questions")

                processed_count += 1
                logger.info(f"Completed processing document {processed_count} of {topics_num}")
            except StopIteration:
                # No more documents to process
                logger.warning(f"No more documents available after processing {processed_count} documents")
                break
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}", exc_info=True)
                raise

        logger.info(f"Finished processing {processed_count} topics")

    @abstractmethod
    def subtopic_extraction_node(self, state: State) -> SubtopicExtractionNodeResponse:
        """
        Extract subtopics from the syllabus document.

        Args:
            state: Current state of the graph

        Returns:
            Updated state with extracted subtopics

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subtopic extraction node is not implemented")

    @abstractmethod
    def question_planning_node(self, state: State) -> PlanningNodeResponse:
        """
        Plan the questions to be generated for each subtopic.

        Args:
            state: Current state of the graph

        Returns:
            Updated state with a question plan

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Question planning node is not implemented")

    @abstractmethod
    def batch_selection_node(self, state: State) -> BatchSelectionNodeResponse:
        """
        Select the next batch of questions to be generated.

        Args:
            state: Current state of the graph

        Returns:
            Updated state with the selected batch

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Batch selection node is not implemented")

    @abstractmethod
    def batch_question_generation_node(self, state: State) -> State:
        """
        Generate questions for the current batch.

        Args:
            state: Current state of the graph

        Returns:
            Updated state with generated questions

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Batch question generation node is not implemented")

    @abstractmethod
    def question_saving_node(self, state: State) -> State:
        """
        Save the generated questions to the state.

        Args:
            state: Current state of the graph

        Returns:
            Updated state with saved questions

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Question saving node is not implemented")

    @abstractmethod
    def batch_decision_node(self, state: State) -> str:
        """
        Decide whether to process the next batch or end processing.

        Args:
            state: Current state of the graph

        Returns:
            Decision string: 'next_batch' or 'end'

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Batch decision node is not implemented")