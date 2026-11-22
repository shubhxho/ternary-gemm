# Security policy

Please report vulnerabilities privately via GitHub's "Report a vulnerability" (Security tab)
rather than a public issue.

Scope notes: the safe API validates every buffer length and shape before a kernel launch, so an
out-of-bounds device access reachable from safe Rust is a bug worth reporting. The experimental
`PersistentEngine` shares mapped host memory with a running kernel by design; see its docs.
