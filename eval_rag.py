from hybrid_router import load_vectorstore
from rag_demo import ask_question

CASES = [
    ("What is the order number of the 3W050 angle head?", "0 608 810 043", "rexroth_tightening_spindles.pdf"),
    ("What is the max torque of the 3W090 angle head?", "90", "rexroth_tightening_spindles.pdf"),
    ("What is the weight of the 3GE67 planetary gearbox?", "0.5", "rexroth_tightening_spindles.pdf"),
    ("What is the order number of the EC303 EC motor?", "0 608 701 017", "rexroth_tightening_spindles.pdf"),
    ("What is the installation length of the 3AR redundant adapter?", "57", "rexroth_tightening_spindles.pdf"),
    ("What is the reduction of the 3GE27 planetary gearbox?", "27", "rexroth_tightening_spindles.pdf"),
    ("What is the order number of the 3S2M8 feed output drive?", "0 608 800 649", "rexroth_tightening_spindles.pdf"),
    ("What is the min circle diameter for 3 spindles of type 3W050?", "40", "rexroth_tightening_spindles.pdf"),
    ("What is the working range of the size 3 feed output drive?", "53", "rexroth_tightening_spindles.pdf"),
    ("What does error E-451 mean?", "torque sensor", "maintenance_manual.txt"),
]

vs = load_vectorstore()
answer_hits = source_hits = 0

for q, expected, source in CASES:
    answer, docs = ask_question(vs, q)
    a_ok = expected.replace(" ", "").lower() in answer.replace(" ", "").lower()
    s_ok = any(d.metadata["source"] == source for d in docs)
    answer_hits += a_ok
    source_hits += s_ok
    print(f"{'OK ' if a_ok else 'BAD'} answer | {'OK ' if s_ok else 'BAD'} source | {q}")

n = len(CASES)
print(f"\nAnswer accuracy: {answer_hits}/{n}   Retrieval hit rate: {source_hits}/{n}")