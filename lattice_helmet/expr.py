"""A small, safe evaluator for field expressions such as

    smoothstep(40, 0, dist(x, y, z, 0, 0, 125))

Only arithmetic, comparisons, numbers, the variables passed in and the
functions below are allowed, so recipes from other people (or an Onshape
panel) cannot run arbitrary code.
"""

import ast
import operator

import numpy as np


def _smoothstep(e0, e1, t):
    s = np.clip((t - e0) / (e1 - e0), 0.0, 1.0)
    return s * s * (3 - 2 * s)


FUNCTIONS = {
    "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "atan2": np.arctan2,
    "min": np.minimum, "max": np.maximum, "clip": np.clip,
    "lerp": lambda a, b, t: a + (b - a) * t,
    "smoothstep": _smoothstep,
    "dist": lambda x, y, z, px, py, pz: np.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2),
    "where": np.where,
}
CONSTANTS = {"pi": np.pi}

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_CMP = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


class ExprError(ValueError):
    pass


def compile_expr(text):
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"bad expression: {e.msg}") from None
    _check(tree.body)
    return tree.body


def names_used(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _check(node):
    for n in ast.walk(node):
        ok = isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call,
                            ast.Name, ast.Constant, ast.Load, ast.IfExp,
                            *_BIN, *_CMP, *_UN))
        if not ok:
            raise ExprError(f"not allowed in expressions: {type(n).__name__}")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS):
            raise ExprError("only these functions are allowed: " + ", ".join(sorted(FUNCTIONS)))
        if isinstance(n, ast.Call) and n.keywords:
            raise ExprError("keyword arguments are not allowed")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            raise ExprError("only numeric constants are allowed")


def evaluate(node, env):
    if isinstance(node, ast.Constant):
        return float(node.value)  # floats overflow instead of building huge ints
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ExprError(f"unknown name {node.id!r}; available: {', '.join(sorted(env))}")
    if isinstance(node, ast.BinOp):
        return _BIN[type(node.op)](evaluate(node.left, env), evaluate(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UN[type(node.op)](evaluate(node.operand, env))
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ExprError("chained comparisons are not allowed")
        return _CMP[type(node.ops[0])](evaluate(node.left, env), evaluate(node.comparators[0], env))
    if isinstance(node, ast.IfExp):
        return np.where(evaluate(node.test, env), evaluate(node.body, env), evaluate(node.orelse, env))
    if isinstance(node, ast.Call):
        return FUNCTIONS[node.func.id](*[evaluate(a, env) for a in node.args])
    raise ExprError(f"not allowed: {type(node).__name__}")
