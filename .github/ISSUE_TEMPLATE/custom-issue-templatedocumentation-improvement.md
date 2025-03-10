---
name: Custom issue templateDocumentation Improvement
about: Suggest improvements to PACF Framework documentation
title: ''
labels: documentation
assignees: ''

---

name: Documentation Improvement
description: Suggest improvements to PACF Framework documentation
title: "[DOCS] "
labels: ["documentation"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for helping us improve the PACF Framework documentation!
  - type: dropdown
    id: doc_type
    attributes:
      label: Documentation Type
      description: What type of documentation needs improvement?
      options:
        - README
        - Code Comments/Docstrings
        - Function/API Documentation
        - Examples
        - Installation Instructions
        - Usage Instructions
        - Other (specify below)
    validations:
      required: true
  - type: input
    id: location
    attributes:
      label: Location
      description: Where is the documentation that needs improvement? (file path, function name, etc.)
      placeholder: e.g., README.md or PACF_v1.py function detect_patterns()
    validations:
      required: true
  - type: textarea
    id: current
    attributes:
      label: Current Documentation
      description: What does the current documentation say?
      placeholder: Copy the relevant section here...
    validations:
      required: false
  - type: textarea
    id: improvement
    attributes:
      label: Suggested Improvement
      description: How should the documentation be improved?
      placeholder: I suggest changing/adding...
    validations:
      required: true
  - type: textarea
    id: reason
    attributes:
      label: Reason for Change
      description: Why is this change needed?
      placeholder: This change would help users understand...
    validations:
      required: false
  - type: textarea
    id: additional
    attributes:
      label: Additional Context
      description: Add any other context about the documentation improvement.
