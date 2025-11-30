# Style Guide and Code Quality Configuration

This document describes the style checker and code quality tools configured for the stock-report-generator project.

## Configuration Files

The project includes the following style checker configurations:

### 1. Flake8 (`.flake8`)
- **Purpose**: Linting for Python code style and errors
- **Configuration**: 
  - Max line length: 120 characters
  - Max complexity: 10
  - Excludes test files, data directories, and build artifacts

**Usage:**
```bash
pip install flake8
flake8 src/
```

### 2. Pylint (`.pylintrc`)
- **Purpose**: Comprehensive static code analysis
- **Configuration**:
  - Max line length: 120 characters
  - Relaxed docstring requirements
  - Customized complexity limits
  - Excludes data and test directories

**Usage:**
```bash
pip install pylint
pylint src/
```

### 3. Ruff (`pyproject.toml` - `[tool.ruff]`)
- **Purpose**: Fast, modern Python linter (alternative to Flake8)
- **Configuration**:
  - Line length: 120
  - Selects multiple rule sets (pycodestyle, Pyflakes, isort, etc.)
  - Per-file ignores for `__init__.py` and tests

**Usage:**
```bash
pip install ruff
ruff check src/
ruff format src/
```

### 4. Black (`pyproject.toml` - `[tool.black]`)
- **Purpose**: Uncompromising Python code formatter
- **Configuration**:
  - Line length: 120
  - Target Python versions: 3.10, 3.11, 3.12

**Usage:**
```bash
pip install black
black src/
```

### 5. isort (`pyproject.toml` - `[tool.isort]`)
- **Purpose**: Import statement sorting
- **Configuration**:
  - Compatible with Black
  - Line length: 120

**Usage:**
```bash
pip install isort
isort src/
```

### 6. MyPy (`pyproject.toml` - `[tool.mypy]`)
- **Purpose**: Static type checking
- **Configuration**:
  - Python version: 3.10+
  - Ignores missing imports for external libraries
  - Relaxed settings for gradual typing adoption

**Usage:**
```bash
pip install mypy
mypy src/
```

### 7. EditorConfig (`.editorconfig`)
- **Purpose**: Maintain consistent coding styles across editors
- **Configuration**:
  - UTF-8 encoding
  - LF line endings
  - 4-space indentation for Python
  - 120 character line length

## Recommended Development Setup

### Install all style checkers:
```bash
pip install flake8 pylint ruff black isort mypy
```

### Pre-commit checks (manual):
```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
ruff check src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

## CI/CD Integration

These configurations can be used in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Lint with Flake8
  run: flake8 src/ tests/

- name: Check with Ruff
  run: ruff check src/ tests/

- name: Format check with Black
  run: black --check src/ tests/

- name: Type check with MyPy
  run: mypy src/
```

## Exclusions

The following directories/files are excluded from style checking:
- `__pycache__/`
- `.venv/`, `venv/`, `env/`
- `build/`, `dist/`
- `.pytest_cache/`, `.mypy_cache/`
- `reports/`
- `data/processed/`, `data/raw/`, `data/outputs/`
- `*.egg-info/`

## Code Style Standards

- **Line Length**: 120 characters
- **Indentation**: 4 spaces
- **Import Style**: isort with Black compatibility
- **String Quotes**: Double quotes preferred (Black default)
- **Docstrings**: Recommended but not strictly required for private methods

## Exceptions

Some common exceptions configured:
- Missing docstrings allowed for `__init__.py` files
- Line length exceptions for URLs and long strings
- Type checking relaxed for external library imports
- Complexity limits adjusted for agent implementations

## Integration with IDEs

### VS Code
Install extensions:
- Python
- Pylint
- Black Formatter
- Ruff

### PyCharm
- Enable inspections: Settings → Editor → Inspections → Python
- Configure Black as external tool
- Set Ruff as external tool

## Notes

- Ruff is faster than Flake8 and can replace it
- Black ensures consistent formatting across the codebase
- MyPy is optional but recommended for type safety
- Pylint provides the most comprehensive analysis but can be slower


