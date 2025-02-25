# Contributing to DocMind AI

Thank you for considering contributing to DocMind AI! This document outlines the process for contributing to the project and how to get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it to understand the expectations we have for everyone who contributes to this project.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR-USERNAME/docmind-ai.git
   cd docmind-ai
   ```

3. Set up the development environment:

   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. Set up the pre-commit hooks (if available):

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

1. Create a branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-you-are-fixing
   ```

2. Make your changes and commit them using meaningful commit messages:

   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

3. Update documentation if necessary, including comments, docstrings, and README updates.
4. Run tests to ensure your changes don't break existing functionality.
5. Push your branch to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request from your fork to the main repository.

## Pull Request Process

- Update the README.md with details of changes if applicable.
- Ensure all tests pass and add new tests for new functionality.
- Make sure your code follows the project's coding standards.
- Update the documentation with any necessary changes.
- The PR needs approval from at least one maintainer before merging.
- Squash your commits into a single meaningful commit if possible.

## Coding Standards

- Follow PEP 8 style guidelines for Python code.
- Use meaningful variable, function, and class names.
- Include docstrings for all functions, classes, and modules.
- Keep functions focused on a single responsibility.
- Limit line length to 100 characters.
- Use type hints where appropriate.

## Testing

- All new features should include appropriate tests.
- Run the existing test suite before submitting a PR:

  ```bash
  pytest
  ```

- Aim for high test coverage for new code.

## Documentation

- Update documentation when adding or modifying features.
- Include docstrings that follow the Google Python Style Guide.
- Update the README.md if your changes affect usage or installation.
- Add example code when introducing new features.

## Issue Reporting

When reporting issues, please use the provided issue templates if available and include:

- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Screenshots or logs if applicable

## Feature Requests

Feature requests are welcome! Please provide:

- A clear description of the feature
- The motivation for adding this feature
- Examples of how the feature would be used
- Any additional context or screenshots

Thank you for contributing to DocMind AI! Your efforts help make this project better for everyone.
