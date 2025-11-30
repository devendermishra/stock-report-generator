# Test Suite

This directory contains unit tests for the stock report generator project.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_summarizer_parsers.py      # Tests for JSON parsing utilities
│   ├── test_summarizer_prompts.py      # Tests for prompt building utilities
│   └── test_report_formatter_utils.py  # Tests for report formatting utilities
├── integration/       # Integration tests for end-to-end workflows
│   └── test_full_workflow.py  # Full workflow integration tests
├── fixtures/          # Test fixtures and mock data
├── conftest.py        # Pytest configuration
└── __init__.py
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/unit/test_summarizer_parsers.py
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Coverage

Currently, the following utilities are covered by tests:

1. **Summarizer Parsers** (`test_summarizer_parsers.py`)
   - JSON response parsing
   - Fallback behavior for invalid JSON
   - Multiline JSON handling

2. **Summarizer Prompts** (`test_summarizer_prompts.py`)
   - Prompt building for summarization
   - Prompt building for insight extraction
   - Document chunk prompt generation

3. **Report Formatter Utils** (`test_report_formatter_utils.py`)
   - Market cap formatting
   - Trends list formatting
   - Risk list formatting
   - Scoring functions (sector outlook, financial performance, management quality)

4. **Integration Tests** (`test_full_workflow.py`)
   - End-to-end workflow testing
   - AI mode workflow validation
   - Structured mode workflow validation
   - Error handling and edge cases
   - Symbol validation and normalization
   - Auto-population of company information

## Adding New Tests

When adding new tests, follow the naming convention:
- `test_<module_name>.py` for unit tests
- `test_<feature>_<workflow>.py` for integration tests
- Place unit tests in `tests/unit/` directory
- Place integration tests in `tests/integration/` directory

Test classes should be named `Test<FunctionName>` or `Test<FeatureName>` and test methods should be named `test_<scenario>`.

**Integration Test Guidelines:**
- Mock external API calls (OpenAI, yfinance, etc.)
- Test complete workflows end-to-end
- Verify data flow between components
- Test both success and error scenarios
- Use fixtures for reusable test data

## Requirements

Install pytest if not already installed:
```bash
pip install pytest pytest-cov
```


