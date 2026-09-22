"""A thin provider layer for forced-tool-call completions.

One protocol, several vendors, over ``httpx`` -- which is already a core
dependency, so candidate generation needs no vendor SDK and no optional extra.

Two rules shape this package, and both come from the harness rather than from
taste:

**Selection is explicit and failure is loud.** There is deliberately no
fallback chain. Every candidate records the model that produced it, and a
chain would let a rerun quietly produce a different mixture of models with
nobody having chosen it -- exactly the kind of unexamined variation this
project exists to rule out.

**The model id recorded is the one the response reports**, never the alias
that was requested, so a floating name can never reach a candidate's
provenance.
"""
