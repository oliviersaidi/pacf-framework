---
name: Feature Request
about: Suggest a new feature for PACF Framework
title: ''
labels: enhancement
assignees: ''

---

name: Feature Request
description: Suggest a new feature for PACF Framework
title: "[FEATURE] "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature for PACF Framework!
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear description of the problem or limitation you're experiencing.
      placeholder: I'm frustrated when [...]
    validations:
      required: false
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear description of what you want to happen or the feature you'd like to see.
      placeholder: I would like PACF to [...]
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: Any alternative solutions or features you've considered.
      placeholder: I've thought about implementing it by [...]
    validations:
      required: false
  - type: dropdown
    id: algorithm_type
    attributes:
      label: Related Algorithm Type
      description: What type of algorithm or pattern is this related to?
      multiple: true
      options:
        - TSP Solver
        - Pattern Detection
        - Clustering
        - Meta-Learning
        - Visualization
        - Other (specify in additional context)
    validations:
      required: false
  - type: textarea
    id: academic
    attributes:
      label: Academic Considerations
      description: If applicable, describe how this feature aligns with the research paper.
      placeholder: This feature would extend the framework by [...]
    validations:
      required: false
  - type: textarea
    id: additional
    attributes:
      label: Additional Context
      description: Add any other context, code examples, or references about the feature request here.
