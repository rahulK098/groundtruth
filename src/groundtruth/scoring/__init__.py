"""Retrieval scoring: span -> chunk resolution, metrics, and aggregation.

Pure logic, no I/O. Every function here takes a ranked list and a set of
labels and returns a number -- nothing here reads a file, calls a model, or
knows what a retrieval configuration is. That is what makes it independently
testable against worked arithmetic (ADR-0004), and it is why the gate
(`groundtruth.gate`) can treat this module as a black box it trusts.
"""
