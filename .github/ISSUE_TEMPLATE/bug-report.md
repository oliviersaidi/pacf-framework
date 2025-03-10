---
name: Bug Report
about: Report a bug in PACF Framework
title: ''
labels: bug
assignees: ''

---

name: Bug Report
description: Report a bug in PACF Framework
title: "[BUG] "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: version
    attributes:
      label: PACF Version
      description: What version of PACF are you using?
      placeholder: e.g., v1.0
    validations:
      required: true
  - type: dropdown
    id: os
    attributes:
      label: Operating System
      description: What operating system are you using?
      options:
        - macOS
        - Windows
        - Linux
        - Other (specify in environment details)
    validations:
      required: true
  - type: input
    id: python-version
    attributes:
      label: Python Version
      description: What Python version are you using?
      placeholder: e.g., Python 3.11
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment Details
      description: Please provide any additional environment details (installed packages, etc.)
      value: |
        ```
        Optional: Output of pip list or conda list
        ```
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Describe the bug and what you expected to happen.
      placeholder: A clear and concise description of what the bug is.
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior
      placeholder: |
        1. Run command '...'
        2. Load data '...'
        3. Execute function '...'
        4. See error
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant Log Output
      description: Please copy and paste any relevant log output. This will be automatically formatted into code.
      render: shell
  - type: textarea
    id: additional
    attributes:
      label: Additional Context
      description: Add any other context about the problem here.
