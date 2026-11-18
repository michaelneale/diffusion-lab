"""Shared task set and checks for the diffusion-vs-Skippy WAN test.

Selected tasks are ones #3124 reported few-pass successes on; held-out tasks were
not used to choose settings. Checks are automatic, except code, which is saved
for review and executed offline after inspection.
"""
import json, re

TASKS = {
  # selected in #3124
  'extract': ('selected', 'Return only a JSON object with keys name, count, enabled. Facts: the name is cedar, the count is 17, and enabled is false. Do not add other keys or prose.'),
  'reason': ('selected', 'A shop starts with 23 apples, sells 7, receives 12, then packs all remaining apples equally into 4 boxes. How many apples are in each box? Show the arithmetic briefly and give the final answer.'),
  'travel': ('selected', 'A train travels 120 km at 60 km/h, stops for 15 minutes, then travels 45 km at 90 km/h. What is the total elapsed time in minutes? Show the arithmetic in at most 60 words.'),
  'brackets': ('selected', 'Write Python function balanced(s) that returns True exactly when round parentheses in s are balanced. Ignore all other characters. A closing parenthesis before its matching opener is invalid. Return only the function, with no examples or prose.'),
  'list32': ('selected', 'Write exactly 32 lines, no heading, no fences, no explanation. For each integer 1 through 32, write the line "Item N: status pending review." replacing N with that integer. Include all 32 items in order.'),
  # held out: not used to pick settings
  'boxes': ('heldout', 'A warehouse has 48 boxes. It ships three quarters of them, then receives 18 more boxes. It divides all boxes now present equally between 5 trucks. How many boxes go on each truck? Show the arithmetic and final answer in at most 60 words.'),
  'discount': ('heldout', 'A customer buys 3 notebooks at $17 each and 2 pens at $9 each. A 20% discount applies to the whole purchase. They pay with $100. What change should they receive? Show the arithmetic in at most 60 words.'),
  'gcd': ('heldout', 'Write Python function gcd(a, b) using the Euclidean algorithm. Handle negative inputs and return 0 for gcd(0, 0). No imports. Return only the function, no examples or prose.'),
  'json2': ('heldout', 'Return only a JSON object with keys city, population, coastal. Facts: the city is Hobart, the population is 251000, and coastal is true. Do not add other keys or prose.'),
  'list20': ('heldout', 'Write exactly 20 lines, no heading, no fences, no explanation. For each integer 1 through 20, write the line "Node N: online." replacing N with that integer. Include all 20 lines in order.'),
}
CANVAS = {'list32': 512, 'list20': 256}  # everything else: 128

def last_number(text):
    nums = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    return float(nums[-1]) if nums else None

def strict_json(text, want):
    try:
        got = json.loads(text.strip(), object_pairs_hook=lambda kv: dict(kv) if len({k for k, _ in kv}) == len(kv) else None)
    except ValueError:
        return False
    return isinstance(got, dict) and set(got) == set(want) and all(type(got[k]) is type(v) and got[k] == v for k, v in want.items())

def check(task, text):
    """True/False, or None when human review is required (code)."""
    t = text.strip()
    if task == 'extract': return strict_json(t, {'name': 'cedar', 'count': 17, 'enabled': False})
    if task == 'json2': return strict_json(t, {'city': 'Hobart', 'population': 251000, 'coastal': True})
    if task == 'reason': return last_number(t) == 7
    if task == 'travel': return last_number(t) == 165
    if task == 'boxes': return last_number(t) == 6
    if task == 'discount': return last_number(t) == 44.8
    if task == 'list32': return t == '\n'.join(f'Item {i}: status pending review.' for i in range(1, 33))
    if task == 'list20': return t == '\n'.join(f'Node {i}: online.' for i in range(1, 21))
    return None
