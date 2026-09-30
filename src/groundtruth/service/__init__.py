"""The FastAPI service: a thin adapter over the same Retriever the gate measures.

ADR-0008: nothing in the core library imports FastAPI, and this package holds
no retrieval logic of its own -- routes validate input, dispatch the shared
``Retriever`` off the event loop, and serialize what it returns. A test
imports the core in a clean interpreter and asserts ``fastapi`` never loads.
"""
