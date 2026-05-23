# Development Setup Guide

This document explains how to set up the development environment for the Guided Label Propagation library with all quality assurance tools configured.

## Prerequisites

- Python 3.9 or higher
- Git
- pip or conda

## Installation

### 1. Install Development Dependencies

```bash
# Core development tools
pip install ruff mypy black isort pytest pytest-cov pre-commit

# Scientific computing dependencies (if not already installed)
pip install networkit polars numpy scipy scikit-learn

# Additional testing tools
pip install pytest-xdist pytest-timeout pytest-benchmark pytest-memray pytest-mock hypothesis

# Documentation tools
pip install pydocstyle bandit

# YAML validation
pip install pyyaml
```

### 2. Install Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Test the installation
pre-commit run --all-files
```

## Configuration Files Overview

### `.ruff.toml` - Code Linting
- **Purpose**: Comprehensive linting with scientific computing optimizations
- **Features**: 
  - 90+ rule categories enabled
  - Scientific library import conventions enforced
  - Complexity limits adjusted for algorithms
  - Per-directory customization (tests, examples, source)

**Usage:**
```bash
# Check all code
ruff check src/ tests/

# Auto-fix issues
ruff check src/ tests/ --fix

# Format code
ruff format src/ tests/
```

### `.mypy.ini` - Type Checking
- **Purpose**: Static type checking with C++ binding accommodations  
- **Features**:
  - Strict typing for core modules
  - NetworkIt C++ binding exceptions
  - Scientific library type stub configuration
  - Relaxed rules for tests and examples

**Usage:**
```bash
# Type check source code
mypy src/

# Check specific module
mypy src/glp/

# Generate type coverage report
mypy src/ --html-report mypy-report/
```

### `pytest.ini` - Test Configuration
- **Purpose**: Comprehensive testing setup for scientific computing
- **Features**:
  - 15+ test markers for organization
  - Coverage reporting with exclusions
  - Parallel test execution
  - Memory and performance monitoring
  - Scientific library warning filters

**Usage:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "network and not slow"
pytest -m "fast"
pytest -m "glp and small_data"

# Run tests in parallel
pytest -n auto
```

### `.pre-commit-config.yaml` - Quality Pipeline
- **Purpose**: Automated code quality enforcement
- **Features**:
  - 13 repository sources with 25+ hooks
  - Scientific computing specific validations
  - Custom hooks for NetworkIt patterns
  - Data validation for CSV fixtures
  - Security scanning with appropriate exclusions

**Usage:**
```bash
# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files

# Update hook versions
pre-commit autoupdate
```

## Test Markers Reference

The pytest configuration defines comprehensive test markers:

### Core Functionality
- `unit` - Unit tests for individual functions
- `integration` - Multi-component integration tests  
- `end_to_end` - Complete workflow tests

### Domain-Specific
- `network` - Network construction and analysis
- `glp` - Guided Label Propagation algorithms
- `timeseries` - Temporal network analysis
- `validation` - Model validation and metrics

### Performance
- `fast` - Tests under 1 second
- `slow` - Tests over 10 seconds
- `memory_intensive` - Tests requiring >1GB RAM
- `cpu_intensive` - CPU-heavy computations

### Data Scale
- `small_data` - <1,000 nodes
- `medium_data` - 1,000-10,000 nodes  
- `large_data` - >10,000 nodes

### Algorithm Types
- `numerical` - Numerical precision tests
- `algorithmic` - Algorithm correctness
- `stochastic` - Random process tests

## Development Workflow

### 1. Code Changes
```bash
# Make your changes
git add .

# Pre-commit hooks run automatically
git commit -m "Your commit message"
```

### 2. Testing
```bash
# Quick tests during development
pytest -m "fast and not external"

# Full test suite before PR
pytest -m "not large_data"

# Performance regression testing
pytest --benchmark-only
```

### 3. Quality Checks
```bash
# Run all quality checks manually
ruff check src/ tests/
mypy src/
pytest --cov=src

# Or use pre-commit
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

**Ruff conflicts with Black:**
- Fixed: Ruff configuration uses Black-compatible settings

**MyPy errors with NetworkIt:**
- Fixed: NetworkIt modules ignore missing imports and errors
- C++ bindings have limited type information

**Pre-commit hooks fail:**
- Install missing dependencies: `pip install bandit pydocstyle`
- Skip problematic hooks: `SKIP=mypy git commit`

**Tests time out:**
- Use smaller test data: `pytest -m "small_data"`
- Increase timeout: `pytest --timeout=600`

### Performance Optimization

**For Large Networks:**
```python
# Use optimized test markers
pytest -m "network and small_data and not cpu_intensive"

# Parallel execution
pytest -n auto

# Memory monitoring
pytest --memray
```

**For CI/CD:**
```bash
# Quick pipeline
pytest -m "fast and not external"

# Standard pipeline  
pytest -m "not memory_intensive and not external"

# Nightly comprehensive
pytest -m "slow or memory_intensive or large_data"
```

## Configuration Customization

### Adjusting Ruff Rules
Edit `.ruff.toml` to modify linting rules:
```toml
[tool.ruff.lint]
ignore = [
    "E501",  # Add custom ignores
    "PLR2004",  # Magic value usage
]
```

### MyPy Strictness
Adjust type checking in `.mypy.ini`:
```ini
[mypy-src.your_module.*]
disallow_untyped_defs = False  # Relax for specific modules
```

### Test Configuration
Modify `pytest.ini` for custom test behavior:
```ini
[tool:pytest]
addopts = 
    -v
    --cov=src
    --timeout=600  # Increase timeout
```

## Scientific Computing Considerations

### NetworkIt C++ Bindings
- Type hints limited due to C++ nature
- Some MyPy errors expected and ignored
- Performance-critical code may bypass some checks

### Large Dataset Testing
- Use appropriate test markers
- Monitor memory usage with pytest-memray
- Consider test data size limits

### Numerical Precision
- Float comparison tolerances configured
- Hypothesis testing for property-based tests
- Scientific warning filters in place

## Getting Help

- **Configuration Issues**: Check syntax with Python validation
- **Tool-specific Problems**: Refer to individual tool documentation
- **Scientific Computing**: Check per-module configurations
- **Performance**: Use appropriate test markers and parallel execution

## Continuous Integration

The configuration files are designed to work with CI/CD systems:

```yaml
# Example GitHub Actions usage
- name: Run quality checks
  run: |
    pre-commit run --all-files
    
- name: Run tests  
  run: |
    pytest -m "not memory_intensive and not external" --cov=src
```

This setup ensures consistent code quality while accommodating the unique requirements of scientific computing and network analysis workflows.