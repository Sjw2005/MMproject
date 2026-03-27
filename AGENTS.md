# Project Instructions

## Skill Discovery
- When searching for external skills in this project, prefer `npx skills find <query>`.
- Do not rely on the built-in `skill(...)` tool as the primary discovery path in this repository.
- Reason: `npx skills find` has been verified to work here, while the built-in `skill` tool has repeatedly failed with connection errors.

## Fallback Behavior
- If a session cannot load skills through the built-in tool, switch to `npx skills find <query>` immediately instead of retrying the same failing path many times.
- If a user wants to install a discovered skill, use `npx skills add <owner/repo@skill>`.

## Local Context
- Keep `findings.md` updated when the available skill workflow changes.
- Treat this file as the project-level default instruction for future AI sessions in this repository.
