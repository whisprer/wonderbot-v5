# Phase 11 — External Workspace Actions + Ganglion Collapse Telemetry

Phase 11 extends WonderBot beyond purely internal tools.

## New capabilities

- Workspace file listing, reading, searching, and writing within a configured root.
- Allowlisted local command execution inside the workspace root.
- Diagnostics now expose workspace permissions and ganglion collapse metrics.

## Safety model

The external action layer is intentionally constrained:

- All file paths are confined to `workspace.root`.
- Writes can be disabled independently of reads.
- Command execution is opt-in via `workspace.allow_commands`.
- Commands are checked against `workspace.allowed_commands`.
- Dry-run remains the default for tool execution.

## New tools

- `workspace_list`
- `workspace_read`
- `workspace_search`
- `workspace_write`
- `workspace_run`

## New CLI helpers

- `/workspace [path]`
- `/read <path>`
- `/grep <query> [path=.] [glob=*]`

## Ganglion collapse telemetry

Earlier phases wrote event signatures into the ganglion and evolved a continuous CA field, but did not compute an explicit collapse measure.

Phase 11 adds:

- `field_entropy`: mean binary entropy across the continuous ganglion field
- `collapse_index`: `1 - field_entropy`

That means the collapse signal now comes from the **continuous ganglion state itself**, not from a rolling byte window.
