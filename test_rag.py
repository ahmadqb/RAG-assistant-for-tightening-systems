"""
Checks that RAG retrieves the right documents and refuses out-of-scope questions.
"""
from rag_demo import load_documents, chunk_documents, build_vectorstore, ask_question

vectorstore = build_vectorstore(chunk_documents(load_documents()))

# Test 1: E-451 from the maintenance manual
print("Test 1: Error code E-451")
answer, docs = ask_question(vectorstore, "What does error code E-451 mean?")
assert "e-451" in answer.lower() or "torque sensor" in answer.lower()
assert any(d.metadata["source"] == "maintenance_manual.txt" for d in docs)
print("PASS\n")

# Test 2: LOTO from safety procedures
print("Test 2: LOTO procedure")
answer, docs = ask_question(vectorstore, "What is the lockout/tagout procedure?")
assert "lockout" in answer.lower() or "loto" in answer.lower() or "lock" in answer.lower()
assert any(d.metadata["source"] == "safety_procedures.txt" for d in docs)
print("PASS\n")

# Test 3: Incident report
print("Test 3: MCH-03 incident")
answer, docs = ask_question(vectorstore, "What happened to MCH-03 in September 2024?")
assert "mch-03" in answer.lower() or "hvac" in answer.lower()
assert any(d.metadata["source"] == "incident_reports.txt" for d in docs)
print("PASS\n")

# Test 4: Out-of-scope question should be refused
print("Test 4: Out-of-scope question")
answer, docs = ask_question(vectorstore, "What is the capital of France?")
refusal_markers = ["don't have", "do not have", "not in", "no information", "documentation", "not mentioned", "not provided"]
assert any(m in answer.lower() for m in refusal_markers), f"Unexpected answer: {answer}"
assert "paris" not in answer.lower()
print("PASS\n")

print("ALL RAG TESTS PASSED")