from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field
from typing_extensions import NotRequired


class Subtopic(BaseModel):
    subtopic_name: str
    topic_title: str = ""
    academic_class: str
    subject: str
    learning_objectives: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    assessment_criteria: List[str] = Field(default_factory=list)
    suggested_activities: List[str] = Field(default_factory=list)


class SubtopicsResponse(BaseModel):
    subtopics: List[Subtopic] = Field(
        default_factory=list, description="The list of extracted subtopics"
    )


class QuestionChoice(BaseModel):
    text: str
    is_correct: bool


class QuestionSolution(BaseModel):
    explanation: str
    steps: List[str] = Field(default_factory=list)


class QuestionMetadata(BaseModel):
    created_by: str
    created_at: str
    updated_at: str
    time_estimate: Dict[str, str]


class Question(BaseModel):
    question_id: str
    text: str
    topic: str
    category: str
    academic_class: str
    examination_level: str
    difficulty: str
    tags: List[str] = Field(default_factory=list)
    choices: List[QuestionChoice] = Field(default_factory=list)
    solution: QuestionSolution
    hint: str
    metadata: Optional[QuestionMetadata] = None


class QuestionsResponse(BaseModel):
    questions: List[Question] = Field(
        default_factory=list, description="The list of generated questions"
    )


class PlannedQuestion(BaseModel):
    question_id: str
    topic: str
    subtopic: str
    difficulty: str
    concept_area: str = ""
    status: str = "planned"  # planned, generating, completed


class QuestionPlan(BaseModel):
    planned_questions: List[PlannedQuestion] = Field(
        default_factory=list, description="The list of planned questions"
    )
    total_questions: int = 0


class SubtopicExtractionNodeResponse(TypedDict):
    subtopics: List[Subtopic]


class PlanningNodeResponse(TypedDict):
    question_plan: QuestionPlan


class BatchSelectionNodeResponse(TypedDict):
    current_batch: List[PlannedQuestion]
    plan_position: NotRequired[int]
