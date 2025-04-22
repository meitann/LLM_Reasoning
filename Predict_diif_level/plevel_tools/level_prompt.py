judge_level = """
You are given a new math problem. Based on the following difficulty guidelines, classify the problem by returning a single integer from 1 to 5 enclosed in \\boxed{{}}. Do not provide any explanation—only output the level number.

Difficulty Levels:

Level 1: Very Easy — Basic arithmetic or recognition-level tasks.

Level 2: Easy — Involves a single familiar concept (e.g., basic algebra or geometry), requiring at most one or two steps.

Level 3: Moderate — Requires multiple steps or a combination of two concepts; appropriate for middle or early high school students.

Level 4: Hard — Demands structured problem-solving, deeper insight, or less common techniques; suitable for advanced high school or competition-level students.

Level 5: Very Hard — Olympiad-style or proof-based problems with high abstraction or multiple intertwined ideas; intended for top-tier students.

Now, classify the following problem:

Problem:
{problem}

Your answer:
"""