#!/usr/bin/env python3
# Only run on manually reviewed inputs; AST checks are NOT a security sandbox.
"""Offline checks on reviewed probe outputs; no model or network access."""
import ast
import json
from pathlib import Path
from canvas_experiment import selftest
ROOT = Path(__file__).resolve().parent
selftest()
rows = []
for path in sorted(ROOT.glob('probe32-*-results.json')):
    for r in json.loads(path.read_text())['runs']:
        text = r['text'].strip()
        row = {'source': path.name, 'name': r['name'], 'steps': r['steps'],
               'visible_tokens': r['visible_tokens_before_end'], 'seconds': r['elapsed_s']}
        if r['name'].startswith('extract-'):
            unfenced = text.removeprefix('```json\n').removesuffix('\n```')
            row['content_correct_after_fence_removal'] = json.loads(unfenced) == {'name':'cedar','count':17,'enabled':False}
            row['strict_json_only'] = text == unfenced
        elif r['name'].startswith('code-'):
            code = text.removeprefix('```python\n').split('\n```')[0]
            try:
                tree = ast.parse(code)
                # Only execute the small, reviewed function body, not generated examples/prose.
                funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                assert len(funcs) == 1 and funcs[0].name == 'unique_in_order'
                func = funcs[0]
                assert not func.decorator_list and not func.args.defaults
                assert not any(isinstance(n, (ast.Import, ast.ImportFrom, ast.While, ast.ClassDef)) for n in ast.walk(func))
                assert all(not n.id.startswith('__') for n in ast.walk(func) if isinstance(n, ast.Name))
                assert all(n.attr in ('add', 'append') for n in ast.walk(func) if isinstance(n, ast.Attribute))
                scope = {'__builtins__': {'set':set, 'sorted':sorted}}
                exec(compile(ast.Module(body=[func], type_ignores=[]), '<reviewed-probe-function>', 'exec'), scope)
                row['syntax_valid'] = True
                row['tests'] = []
                for xs, expected in [([], []), ([2,1,2], [2,1]), ([[1],[2],[1]], [[1],[2]])]:
                    try:
                        actual = scope['unique_in_order'](xs)
                        row['tests'].append({'input':xs, 'expected':expected, 'actual':actual, 'pass':actual == expected})
                    except Exception as e:
                        row['tests'].append({'input':xs, 'pass':False, 'error':f'{type(e).__name__}: {e}'})
            except SyntaxError as e:
                row['syntax_valid'] = False
                row['syntax_error'] = str(e)
        else:
            row['assessment'] = 'See report manual review; visible tokens are not a correctness score.'
        rows.append(row)
assert rows, 'No trial records found; supply reviewed experiment results first'
(ROOT/'probe32-validation.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
