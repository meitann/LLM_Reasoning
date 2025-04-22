pl_prompt_0 = """
Here are some examples of problems along with DeepSeek's reasoning process token lengths. You are given a new problem; please briefly assess the difficulty of the new problem(less than 500 tokens) and predict the expected reasoning length that DeepSeek would generate.
Output an integer within \\boxed{{}}, and the number must lie in the range 200 to 16,384 (inclusive).

Examples:
Problem: "A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?"
Difficulty Level: Simple
reasoning_length: 900

Problem: "Find the continuous functions \\( f: \\mathbb R \\rightarrow \\mathbb R \\) such that for all \\( x \\) and \\( y \\),\n\n\\[ f\\left(\\frac{{x+y}}{{2}}\\right) = \\frac{{f(x) + f(y)}}{{2}} \\]"
Difficulty Level: Easy
Reasoning Length: 2119

Problem: "Using the digits 0, 1, 2, 3, and 4, find the number of 13-digit sequences that can be written such that the difference between any two consecutive digits is 1. Examples of such 13-digit sequences are 0123432123432, 2323432321234, and 3210101234323."
Difficulty Level: Medium
Reasoning Length: 4522

Problem: "99 cards each have a label chosen from 1, 2, ..., 99, such that no non-empty subset of the cards has labels with a total divisible by 100. Show that the labels must all be equal."
Difficulty Level: Hard
Reasoning Length: 8196

Problem: "Assume there is a gathering with $n$ couples attending. Each individual during a certain period can either rest alone or join a chatting group, referred to as a team. Each couple is not in the same team, and every other pair of individuals is in exactly one team together. Prove that when $n \\geq 4$, the total number of teams $k \\geq 2n$"
Difficulty Level: Very Hard
Reasoning Length: 16384

New Problem: {new_problem}
Please predict its reasoning length: 
"""