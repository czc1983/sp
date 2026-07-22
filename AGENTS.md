# Project Collaboration Rules

## Parallel Agent Preference

For this project, treat parallel agent usage as a standing user preference.

The main agent should default to acting as the coordinator/controller for non-trivial work: estimate the effort, judge whether the task can be split, assign concrete code reading, implementation, and verification slices to sub-agents when that will help, then integrate the results and make the final decision.

Use sub-agents when the work can be divided into independent reading, implementation, or verification slices with disjoint write scopes. Long-running tasks may be split across multiple rounds and multiple sub-agents, with each round narrowing the next step. Keep the work local when the task is small, tightly coupled, or would be slower to coordinate than to complete directly.

When a change is highly coupled in the same file or same small code path, do not let multiple agents edit it in parallel. Prefer one implementation agent, plus any number of read-only review or verification agents.

This preference does not override higher-priority safety, tool, or user instructions. It means: actively consider multi-agent execution by default, and use it whenever it materially speeds up the work without increasing risk.

## Mode 1 / Mode 2 Boundary

Mode 2 must remain independent from Mode 1. It may reuse ideas and small utility patterns from Mode 1, but should not import or depend on Mode 1's transfer/render workflow. When sharing concepts such as role anchors or timeline annotation, prefer Mode-2-specific functions, endpoints, and storage fields unless a small pure helper is clearly safe.
