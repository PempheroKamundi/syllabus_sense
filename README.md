# Syllabus Sense - AI Question Generator

An AI-powered tool that analyzes educational syllabi and automatically generates multiple-choice questions for assessment.

## Built for Speed and Scalability

Syllabus Sense is designed with efficiency and scalability at its core:

- **Topic-Based Chunking**: The system intelligently chunks syllabus documents into manageable topics, processing each independently to prevent memory overflow and enable parallel processing
- **Streaming Document Parser**: The document parser implements the iterator pattern, efficiently streaming syllabus content topic-by-topic rather than loading the entire document into memory
- **Batch Question Generation**: Questions are generated in configurable batches within each topic, optimizing API usage and allowing for graceful handling of rate limits
- **Scale-Ready Architecture**: The underlying graph-based workflow is designed to scale infinitely once the planned async processing and Celery-based distributed computing features are implemented

### Code Highlights

#### Streaming Document Parser

The document parser uses the iterator pattern to efficiently stream content:

```python
class BaseSyllabusParser(ABC):
    """Abstract base class for Syllabus parsers."""

    def __iter__(self):
        """Make SyllabusParser an iterator"""
        raise NotImplementedError("SyllabusParser must implement __iter__")

    def __next__(self) -> SyllabusTopic:
        """Get the next topic"""
        raise NotImplementedError("SyllabusParser must implement __next__")

class NormalSyllabusParser(BaseSyllabusParser):
    # ...
    
    def __iter__(self):
        """Make SyllabusParser an iterator"""
        self._generator = self._process_topics()
        return self
        
    def __next__(self) -> SyllabusTopic:
        """Get the next topic, one at a time"""
        if self._generator is None:
            self._generator = self._process_topics()

        title, elements = next(self._generator)
        return SyllabusTopic(title=title, elements=elements)
```

#### Batch Processing for Efficient API Usage

Questions are generated in configurable batches to optimize resource usage:

```python
def batch_selection_node(self, state: State) -> BatchSelectionNodeResponse:
    """Select the next batch of questions to generate from the plan."""
    # ...
    
    # Determine the end position for this batch
    end_pos = min(start_pos + state.batch_size, len(state.question_plan.planned_questions))

    # Get the current batch of questions
    current_batch = state.question_plan.planned_questions[start_pos:end_pos]
    
    # ...
    return {"current_batch": current_batch, "plan_position": end_pos}
```

### Error Handling Capabilities

The system implements robust error handling throughout the workflow:

#### Parsing and Validation Error Handling

```python
try:
    # Ensure we're parsing a string
    parsed_output = parser.parse(str(response.content))
    
    # Return the question plan
    return {"question_plan": parsed_output}

except (ValidationError, json.JSONDecodeError, OutputParserException) as e:
    # Handle parsing errors
    logger.error(f"Error creating question plan: {str(e)}")
    return {"question_plan": QuestionPlan()}
```

#### File Handling with Graceful Error Recovery

```python
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

    # Continue with saving...

except Exception as e:
    logger.error(f"Error saving questions: {e}")
```

#### Loop Protection and Safeguards

```python
# Add a safeguard to detect if we're stuck in a loop
if hasattr(self, '_last_position') and self._last_position == current_position:
    logger.warning(f"Position hasn't changed from {current_position}. Breaking potential loop.")
    return "end"

self._last_position = current_position
```

This component was extracted from [VirtueEducate](https://github.com/PempheroKamundi#virtueducate), a project focused on creating content-relevant educational questions. Syllabus Sense showcases the course content extraction and question generation capabilities of the main project.

## About VirtuEducate

VirtuEducate is an AI-powered educational platform tackling Malawi's low exam pass rates through personalized learning.

**Key Features**:
- **10,000+ exam-style questions** for JCE & MSCE students
- **AI-driven study paths** based on past paper patterns
- **Real-time progress tracking** and performance insights
- **Instant feedback** with detailed explanations
- **Focus on college readiness & admissions**

The Syllabus Sense component helps power VirtuEducate's extensive question bank by automatically generating high-quality assessment questions aligned with the Malawian curriculum.

## Overview

Syllabus Sense is designed to help educators create high-quality assessment questions directly from curriculum documents. The system:

1. Parses Microsoft Word format syllabus documents
2. Extracts key topics and subtopics
3. Identifies learning objectives and key concepts
4. Generates a plan for comprehensive question coverage
5. Creates batches of multiple-choice questions with explanations

The application specifically targets Malawian educational syllabi but can be adapted to other educational contexts.

In the original VirtueEducate project, this system is enhanced by using Retrieval-Augmented Generation (RAG) to incorporate relevant content from subject textbooks, further improving the quality and accuracy of generated questions.

## Architecture

The project uses a directed graph-based workflow to manage the question generation process:

```
subtopic_extraction → question_planning → batch_selection → batch_question_generation → question_saving
                                               ↑                                              |
                                               └──────────────────────────────────────────────┘
```

### Key Components

- **Document Parser**: Extracts structured content from Word documents
- **Syllabus AI Graph**: Coordinates the workflow of nodes via LangGraph
- **LLM Integration**: Uses OpenAI's GPT-4o models to generate intelligent content

### Files Structure

- `main.py`: Entry point for the application
- `syllabus_ai_graph.py`: Implementation of the AI workflow
- `_base_syllabus_ai_graph_template.py`: Template class defining the graph structure
- `data_types.py`: Pydantic models for data structures
- `document_parser/`: Package for parsing syllabus documents
  - `syllabus_parser.py`: Contains logic for processing Word documents
  - `data_types.py`: Data models for parsed syllabus content

## Prerequisites

- Python 3.8+
- An OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/syllabus-sense.git
   cd syllabus-sense
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.sample .env
   # Edit the .env file and add your OpenAI API key
   ```

## Usage

### Basic Usage

```bash
python main.py
```

This will:
1. Process the default syllabus document (`chemistry_form_1_2.docx`)
2. Extract topics and subtopics
3. Generate questions for each topic
4. Save the questions to `generated_questions.json`

### Custom Usage

You can modify `main.py` to specify different source documents or adjust processing parameters.

Example customization:

```python
if __name__ == "__main__":
    path = Path.cwd() / "my_syllabus.docx"
    parser = NormalSyllabusParser.from_file(
        file_path=path)
    workflow = SyllabusAIGraph(document_parser=parser, subject="biology")
    workflow.process(topics_num=3)  # Process just the first 3 topics
```

## Output

The generated questions are saved to `generated_questions.json` in the following format:

```json
[
  {
    "question_id": "Q001",
    "text": "What is the physical state of matter that has a definite shape and volume?",
    "topic": "States of Matter",
    "category": "Basic Knowledge",
    "academic_class": "Form 1",
    "examination_level": "Secondary School",
    "difficulty": "Easy",
    "tags": ["states of matter", "physical properties"],
    "choices": [
      {"text": "Solid", "is_correct": true},
      {"text": "Liquid", "is_correct": false},
      {"text": "Gas", "is_correct": false},
      {"text": "Plasma", "is_correct": false}
    ],
    "solution": {
      "explanation": "Solids have a definite shape and volume because the particles are closely packed in a regular arrangement.",
      "steps": [
        "Recall the properties of different states of matter",
        "Identify that solids maintain both shape and volume",
        "Compare with liquids (definite volume but not shape) and gases (neither definite volume nor shape)"
      ]
    },
    "hint": "Think about which state of matter doesn't change shape when placed in a container."
  },
  // Additional questions...
]
```

## Logging

The application logs detailed information about the question generation process to both the console and a file called `question_generation.log`.

## Customization

### Modifying Question Templates

To adjust the style or format of generated questions, edit the prompt templates in the `syllabus_ai_graph.py` file.

### Changing Output Format

The output structure is defined by the Pydantic models in `data_types.py`. You can modify these models to change what data is included with each question.

### Supporting Different Document Formats

The base parser is designed for Word documents. To support other formats, create a new parser class that inherits from `BaseSyllabusParser` and implements the necessary methods.

## Troubleshooting

### Common Issues

- **OpenAI API errors**: Ensure your API key is correct and has sufficient quota
- **Document parsing errors**: Check if your syllabus follows the expected format with identifiable topic markers
- **Memory issues**: For large syllabi, consider processing them in smaller segments

### Debug Mode

To enable more detailed logging, modify the logging level in `main.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("question_generation.log"), logging.StreamHandler()],
)
```

## License

[MIT License](LICENSE)

## Current Development Issues

The following issues are currently being addressed:

* Implement asynchronous processing for faster question generation
* Implement Celery for distributed processing of large syllabi
* Improve robustness during question generation with retry logic for Pydantic validation errors
* Add rate limiting checks for API calls
* Add multi-model support for different LLM providers
* Add comprehensive unit tests

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
