from hybrid_router import route_question

cases = [
    ("What does error code E-451 mean?", "DOCS"),
    ("What is the lockout/tagout procedure?", "DOCS"),
    ("What happened to MCH-03 in September 2024?", "DOCS"),
    ("What was the average torque for MCH-01 yesterday?", "SQL"),
    ("How many cycles has MCH-05 run this week?", "SQL"),
    ("Which machine had the most downtime last month?", "SQL"),
]

for question, expected in cases:
    got = route_question(question)
    status = "PASS" if got == expected else "FAIL"
    print(f"{status}: {question} -> {got} (expected {expected})")