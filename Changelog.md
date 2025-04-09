# Changelog

## v0.0.2 - 2025-04-09

### Added

- Introduced BaseOutputManager interface for dependency injection pattern
- FileOutputManager concrete implementation for persistent storage of generated questions
- Separation of concerns through abstraction of output handling from core processing logic
- Directory validation and auto-creation capability for output paths
- Custom exception hierarchy with SyllabusSenseException as base class
- Specific exception types for different error scenarios (InvalidOutputDirectoryError, etc.)
- Robust error handling for file operations

### Changed

- Refactored syllabus processing workflow to support injectable output managers
- Decoupled data persistence from question generation logic
- Improved question saving mechanism with proper error handling
- Enhanced logging with more detailed information about saved questions

## v0.0.1 - 2025-04-09

### Added

-   Initial implementation of Syllabus AI Graph for processing educational syllabi
-   Document parser for Microsoft Word syllabus files with streaming capabilities
-   Subtopic extraction from syllabus content
-   Question planning based on learning objectives and key concepts
-   Batch question generation with multiple-choice format
-   Detailed explanations and hints for generated questions
-   JSON output format for storing generated questions
-   Basic logging system for tracking the generation process
-   Support for configurable batch sizes when generating questions
-   Error handling throughout the workflow