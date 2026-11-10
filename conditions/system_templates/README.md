# System templates

Each named subfolder defines an experimental configuration: prompt layout,
system/user text, exposed controls, and permitted tuning ranges. Control definitions
are shared from `../linguistic_controls/`; they are not copied into system folders.

Users create a new system template here when they need different controls or
bounds. Copy a source template into a new named folder, give it a distinct ID,
and edit that definition while preserving the source. A generation request only
changes exposed values; it does not expand the system template.

The supplied pilots' `system.txt` requests only the generated Finnish text,
without introductory phrases, headings, explanations, or Markdown wrappers.
Length constraints apply to the complete response. This system instruction is
separate from the user prompt's specific constraints.

Experiments save their own prompt snapshots. Updating a source `system.txt`
applies to newly created experiments; existing experiments and submissions keep
their saved prompts. To use the new default with the same conditions, create a
new experiment with `--copy-from` (unless its selection explicitly overrides the
system prompt).

See [the three-tier model](../README.md).
