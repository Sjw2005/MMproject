# Findings

- 2026-03-27: The built-in `skill(...)` tool failed with a connectivity error in this repository. Per `AGENTS.md`, prefer `npx skills find <query>` for skill discovery in later sessions instead of retrying the built-in skill path.
- 2026-03-28: Multimodal setup work now has shared helpers in `ultralytics/data/multimodal.py` and unified experiment definitions in `experiment_config.py`; future sessions should extend those files instead of reintroducing path-replacement logic in individual scripts.
