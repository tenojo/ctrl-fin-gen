# System templates

Each named subfolder defines an experimental configuration: prompt layout,
system/user text, exposed controls, and permitted tuning ranges. Control definitions
are shared from `../linguistic_controls/`; they are not copied into system folders.

Users create a new system template here when they need different controls or
bounds. Copy a source template into a new named folder, give it a distinct ID,
and edit that definition while preserving the source. A generation request only
changes exposed values; it does not expand the system template.

See [the three-tier model](../README.md).
