import json

from datasets import Dataset

DEEPSEEK_R1_SYSTEM_PROMPT = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered
thinking process.
"""

# 138  1492  3681   3816 16384
PREDICT_REASONING_LENGTH = """
Here are some examples of problems along with DeepSeek's reasoning process token lengths. You are given a new problem; please briefly assess the difficulty of the new problem(less than 500 tokens) and predict the expected reasoning length that DeepSeek would generate.
Output an integer within \\boxed{{}}, and the number must lie in the range 200 to 16,384 (inclusive).

Examples:
Problem: "A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?"
Difficulty Level: Simple
reasoning_length: 138

Problem: "Find the continuous functions \\( f: \\mathbb R \\rightarrow \\mathbb R \\) such that for all \\( x \\) and \\( y \\),\n\n\\[ f\\left(\\frac{{x+y}}{{2}}\\right) = \\frac{{f(x) + f(y)}}{{2}} \\]"
Difficulty Level: Easy
Reasoning Length: 1462

Problem: "Using the digits 0, 1, 2, 3, and 4, find the number of 13-digit sequences that can be written such that the difference between any two consecutive digits is 1. Examples of such 13-digit sequences are 0123432123432, 2323432321234, and 3210101234323."
Difficulty Level: Medium
Reasoning Length: 3681

Problem: "99 cards each have a label chosen from 1, 2, ..., 99, such that no non-empty subset of the cards has labels with a total divisible by 100. Show that the labels must all be equal."
Difficulty Level: Hard
Reasoning Length: 3816

Problem: "Assume there is a gathering with $n$ couples attending. Each individual during a certain period can either rest alone or join a chatting group, referred to as a team. Each couple is not in the same team, and every other pair of individuals is in exactly one team together. Prove that when $n \\geq 4$, the total number of teams $k \\geq 2n$"
Difficulty Level: Very Hard
Reasoning Length: 16384

New Problem: {new_problem}
Please predict its reasoning length: 
"""


PREDICT_DIFFICULT_LEVEL = """
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