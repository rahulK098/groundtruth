"""The golden set: query to labeled-span pairs, and the tooling around them.

This is the foundation every metric is computed against. If a label is wrong,
the comparison table, the gate and the final recommendation are wrong in ways
no downstream rigor can detect -- which is why the validators here reject
incoherent records rather than merely malformed ones, and why an LLM is only
ever allowed to *propose* (ADR-0009).
"""
