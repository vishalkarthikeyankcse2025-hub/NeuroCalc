from __future__ import annotations

import math
import re
from fractions import Fraction
from statistics import mean, median, mode, multimode, variance, stdev

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

x, y, z = sp.symbols("x y z")

SAFE_NAMES = {
    "pi": sp.pi,
    "e": sp.E,
    "I": sp.I,
    "oo": sp.oo,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "factorial": sp.factorial,
    "x": x,
    "y": y,
    "z": z,
}

ALLOWED_EXPR = re.compile(r"^[0-9A-Za-z_+\-*/().,\s^=<>!\[\]]+$")


def normalize(text: str) -> str:
    s = str(text).strip().lower()
    replacements = {
        "×": "*", "✕": "*", "÷": "/", "−": "-", "–": "-", "π": "pi",
        "√": "sqrt", "∞": "oo",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)

    s = re.sub(r"\bwhat is\b", "", s)
    s = re.sub(r"\bplease\b", "", s)
    s = re.sub(r"\bcalculate\b", "", s)
    s = re.sub(r"\bcompute\b", "", s)
    s = re.sub(r"\bevaluate\b", "", s)
    s = re.sub(r"\bfind\b", "", s)
    s = re.sub(r"\bthe result of\b", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def number_list(text: str) -> list[float]:
    s = text.replace(",", "")
    return [
        float(v)
        for v in re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?", s)
    ]


def pretty(value, digits: int = 12) -> str:
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            parts = []
            for item in value:
                parts.append(", ".join(f"{k} = {v}" for k, v in item.items()))
            return " ; ".join(parts) if parts else "No solution"
        return ", ".join(pretty(v, digits) for v in value)

    if isinstance(value, dict):
        return ", ".join(f"{k} = {pretty(v, digits)}" for k, v in value.items())

    if isinstance(value, tuple):
        return " ".join(pretty(v, digits) for v in value)

    if isinstance(value, sp.MatrixBase):
        return sp.pretty(value)

    if isinstance(value, sp.Basic):
        if value.is_Integer:
            return str(value)
        if value.is_Rational:
            return f"{value}  ≈  {sp.N(value, digits)}"
        if value.is_real:
            try:
                f = float(value.evalf())
                if math.isfinite(f):
                    return str(int(f)) if f.is_integer() else f"{f:.{digits}g}"
            except Exception:
                pass
        return str(sp.simplify(value))

    try:
        f = float(value)
        if math.isfinite(f):
            return str(int(f)) if f.is_integer() else f"{f:.{digits}g}"
    except Exception:
        pass

    return str(value)


def parse_math_expr(text: str):
    s = normalize(text)
    if not ALLOWED_EXPR.fullmatch(s):
        raise ValueError("The expression contains unsupported characters.")

    # Remove a trailing period and common wording.
    s = s.rstrip(".")
    return parse_expr(
        s,
        local_dict=SAFE_NAMES,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def solve_equation(text: str):
    s = text.lower().replace("→", "=").strip()
    s = re.sub(r"^\s*(solve|solve for x|find x|what is x if)\s*:?", "", s)
    if "=" not in s:
        raise ValueError("Use an equation like 2x + 5 = 15.")

    left, right = s.split("=", 1)
    left = normalize(left)
    right = normalize(right)

    lhs = parse_math_expr(left)
    rhs = parse_math_expr(right)

    vars_found = sorted(
        set(lhs.free_symbols | rhs.free_symbols),
        key=lambda q: str(q),
    )

    if not vars_found:
        return sp.simplify(lhs - rhs), "comparison"

    if len(vars_found) > 3:
        raise ValueError("Please use at most three variables.")

    solutions = sp.solve(sp.Eq(lhs, rhs), vars_found, dict=True)
    return solutions, "equation"


def derivative(expr_text: str, var_name="x", order=1):
    var = sp.Symbol(var_name)
    expr = parse_math_expr(expr_text)
    return sp.diff(expr, var, order)


def integrate(expr_text: str, var_name="x"):
    var = sp.Symbol(var_name)
    expr = parse_math_expr(expr_text)
    return sp.integrate(expr, var)


def definite_integral(expr_text: str, a, b, var_name="x"):
    var = sp.Symbol(var_name)
    expr = parse_math_expr(expr_text)
    a_sym = sp.Rational(str(a)) if isinstance(a, (int, float)) else sp.sympify(a)
    b_sym = sp.Rational(str(b)) if isinstance(b, (int, float)) else sp.sympify(b)
    return sp.integrate(expr, (var, a_sym, b_sym))


def limit(expr_text: str, point: float, var_name="x", direction=None):
    var = sp.Symbol(var_name)
    expr = parse_math_expr(expr_text)
    kwargs = {}
    if direction is not None:
        kwargs["dir"] = direction
    return sp.limit(expr, var, point, **kwargs)


def matrix_from_text(text: str):
    s = text.strip()
    # Accept [[1,2],[3,4]], [1 2; 3 4], or rows separated by semicolons.
    raw = s.replace(" ", "")
    if raw.startswith("[[") and raw.endswith("]]"):
        import ast
        data = ast.literal_eval(raw)
        return sp.Matrix(data)

    if ";" in s:
        rows = []
        for row in s.split(";"):
            vals = [v for v in row.replace(",", " ").split() if v]
            rows.append([sp.sympify(v, locals=SAFE_NAMES) for v in vals])
        return sp.Matrix(rows)

    raise ValueError("Matrix format: [[1,2],[3,4]] or 1 2; 3 4")


def matrix_operation(text: str):
    low = text.lower()
    if "determinant" in low or re.search(r"\bdet\b", low):
        # Parse the first obvious matrix literal.
        m = re.search(r"\[\[.*\]\]", text)
        if not m:
            raise ValueError("Use determinant of [[1,2],[3,4]].")
        M = matrix_from_text(m.group(0))
        return M.det(), "determinant"

    if "inverse" in low:
        m = re.search(r"\[\[.*\]\]", text)
        if not m:
            raise ValueError("Use inverse of [[1,2],[3,4]].")
        M = matrix_from_text(m.group(0))
        return M.inv(), "inverse"

    if "transpose" in low:
        m = re.search(r"\[\[.*\]\]", text)
        if not m:
            raise ValueError("Use transpose of [[1,2],[3,4]].")
        M = matrix_from_text(m.group(0))
        return M.T, "transpose"

    raise ValueError("Supported matrix operations: determinant, inverse, transpose.")


def statistics_calc(text: str):
    nums = number_list(text)
    if len(nums) < 2:
        raise ValueError("Provide at least two numbers.")

    low = text.lower()

    if "median" in low:
        return median(nums)
    if "mode" in low:
        modes = multimode(nums)
        return modes if len(modes) > 1 else modes[0]
    if "variance" in low:
        return variance(nums)
    if "standard deviation" in low or re.search(r"\bsd\b", low):
        return stdev(nums)
    if "maximum" in low or re.search(r"\bmax\b|\blargest\b|\bhighest\b", low):
        return max(nums)
    if "minimum" in low or re.search(r"\bmin\b|\bsmallest\b|\blowest\b", low):
        return min(nums)
    return mean(nums)


def finance_calc(text: str, intent: str):
    n = number_list(text)
    if intent == "simple_interest":
        if len(n) < 3:
            raise ValueError("Need principal, rate, and time.")
        p, r, t = n[:3]
        interest = p * (r / 100) * t
        return {"interest": interest, "amount": p + interest}

    if intent == "compound_interest":
        if len(n) < 3:
            raise ValueError("Need principal, rate, and time.")
        p, r, t = n[:3]
        amount = p * (1 + r / 100) ** t
        return {"interest": amount - p, "amount": amount}

    if intent == "emi":
        if len(n) < 3:
            raise ValueError("Need principal, annual rate, years.")
        p, annual_rate, years = n[:3]
        months = int(round(years * 12))
        monthly = annual_rate / 1200
        if monthly == 0:
            emi = p / months
        else:
            emi = p * monthly * (1 + monthly) ** months / ((1 + monthly) ** months - 1)
        return {
            "emi": emi,
            "months": months,
            "total_payment": emi * months,
            "total_interest": emi * months - p,
        }

    if intent == "cagr":
        if len(n) < 3:
            raise ValueError("Need initial, final, and years.")
        initial, final, years = n[:3]
        if initial <= 0 or final <= 0 or years <= 0:
            raise ValueError("All CAGR values must be positive.")
        rate = (final / initial) ** (1 / years) - 1
        return rate * 100

    raise ValueError("Unsupported finance operation.")


def conversion(text: str):
    s = text.lower().replace(",", "")
    n = number_list(s)
    if not n:
        raise ValueError("No number found.")

    v = n[0]
    tests = [
        (r"km(?:s)?\s+to\s+miles?", v * 0.621371, "miles"),
        (r"miles?\s+to\s+km(?:s)?", v / 0.621371, "km"),
        (r"kg(?:s)?\s+to\s+pounds?", v * 2.20462262185, "pounds"),
        (r"pounds?\s+to\s+kg(?:s)?", v / 2.20462262185, "kg"),
        (r"meters?\s+to\s+feet", v * 3.280839895, "feet"),
        (r"feet\s+to\s+meters?", v / 3.280839895, "meters"),
        (r"celsius|°c.*fahrenheit|fahrenheit", None, None),
    ]

    for pat, result, unit in tests[:-1]:
        if re.search(pat, s):
            return result, unit

    if re.search(r"celsius.*fahrenheit|°c.*°f", s):
        return v * 9 / 5 + 32, "°F"
    if re.search(r"fahrenheit.*celsius|°f.*°c", s):
        return (v - 32) * 5 / 9, "°C"

    raise ValueError("Supported conversions: km/miles, kg/pounds, meters/feet, Celsius/Fahrenheit.")


def calculate(text: str, intent: str | None = None):
    low = text.lower().strip()

    # Hard safety/structure routing.
    if "=" in low and any(c in low for c in "xyz"):
        return {"type": "equation", "value": solve_equation(low)[0]}

    if re.search(r"\bto\s+(miles?|km|kilometers?|kg|kilograms?|pounds?|feet|meters?|fahrenheit|celsius)\b", low):
        value, unit = conversion(low)
        return {"type": "conversion", "value": value, "unit": unit}

    if "derivative" in low or "differentiate" in low:
        # Patterns: derivative of x^2 / differentiate x^2
        expr = re.split(r"derivative of|differentiate", low, maxsplit=1)[-1].strip()
        if not expr:
            raise ValueError("Example: derivative of x^2 + 3x")
        return {"type": "derivative", "value": derivative(expr)}

    if "integral" in low or "integrate" in low:
        expr = re.split(r"integral of|integrate", low, maxsplit=1)[-1].strip()
        # Optional definite form: integrate x^2 from 0 to 2
        m = re.search(r"^(.*?)\s+from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)$", expr)
        if m:
            return {
                "type": "definite_integral",
                "value": definite_integral(m.group(1), m.group(2), m.group(3)),
            }
        return {"type": "integral", "value": integrate(expr)}

    if "limit" in low:
        m = re.search(r"limit of (.+?) as x approaches (-?\d+(?:\.\d+)?)", low)
        if not m:
            raise ValueError("Example: limit of sin(x)/x as x approaches 0")
        return {
            "type": "limit",
            "value": limit(m.group(1), float(m.group(2))),
        }

    if any(k in low for k in ["matrix", "determinant", "inverse", "transpose"]):
        value, operation = matrix_operation(text)
        return {"type": "matrix", "operation": operation, "value": value}

    if any(k in low for k in ["mean", "average", "median", "mode", "variance", "standard deviation", "max", "min", "largest", "smallest"]):
        return {"type": "statistics", "value": statistics_calc(text)}

    if any(k in low for k in ["simple interest", "compound interest", " emi", "emi ", "cagr"]):
        chosen = intent or (
            "compound_interest" if "compound" in low else
            "simple_interest" if "simple" in low else
            "cagr" if "cagr" in low else
            "emi"
        )
        return {"type": "finance", "intent": chosen, "value": finance_calc(text, chosen)}

    # Trigonometry in degrees.
    if any(k in low for k in ["sin ", "cos ", "tan ", "sine ", "cosine ", "tangent "]):
        n = number_list(low)
        if not n:
            raise ValueError("Provide an angle, e.g. sin 30 degrees.")
        a = math.radians(n[0])
        if "tan" in low or "tangent" in low:
            value = math.tan(a)
        elif "cos" in low or "cosine" in low:
            value = math.cos(a)
        else:
            value = math.sin(a)
        return {"type": "trigonometry", "value": value}

    if "factorial" in low or re.search(r"\d+\s*!\s*$", low):
        n = number_list(low)
        if not n or n[0] < 0 or n[0] != int(n[0]):
            raise ValueError("Factorial requires a non-negative integer.")
        return {"type": "factorial", "value": math.factorial(int(n[0]))}

    m_log10 = re.search(r"\blog10\s*\(?\s*(-?\d+(?:\.\d+)?)", low)
    if m_log10:
        value = float(m_log10.group(1))
        if value <= 0:
            raise ValueError("Log requires a positive value.")
        return {"type": "logarithm", "value": math.log10(value)}

    if re.search(r"\bln\b|natural log", low):
        n = number_list(low)
        if not n or n[0] <= 0: raise ValueError("ln requires a positive value.")
        return {"type": "logarithm", "value": math.log(n[0])}

    if re.search(r"\bsqrt\b|square root|√", low):
        n = number_list(low)
        if not n or n[0] < 0: raise ValueError("Square root requires a non-negative number.")
        return {"type": "square_root", "value": math.sqrt(n[0])}

    if "%" in low:
        n = number_list(low)
        if len(n) < 2: raise ValueError("Need percentage and base value.")
        if "increase" in low or "increase by" in low:
            return {"type": "percentage_increase", "value": n[1] * (1 + n[0] / 100)}
        if "decrease" in low or "decrease by" in low or "reduce" in low:
            return {"type": "percentage_decrease", "value": n[1] * (1 - n[0] / 100)}
        return {"type": "percentage", "value": n[0] * n[1] / 100}

    # Basic arithmetic and generic exact expressions.
    expr = normalize(low)
    # Handle natural-language arithmetic if words survived unexpectedly.
    expr = re.sub(r"\bplus\b", "+", expr)
    expr = re.sub(r"\bminus\b", "-", expr)
    expr = re.sub(r"\btimes\b", "*", expr)
    expr = re.sub(r"\bdivided by\b", "/", expr)
    expr = re.sub(r"\bmultiplied by\b", "*", expr)
    expr = re.sub(r"(?<=\d)\s*x\b", "* x", expr)

    value = parse_math_expr(expr)
    return {"type": "expression", "value": sp.simplify(value)}


def explain(result: dict) -> str:
    t = result["type"]
    v = result.get("value")

    if t == "percentage":
        return "percentage = rate ÷ 100 × base"
    if t == "percentage_increase":
        return "new value = base × (1 + rate ÷ 100)"
    if t == "percentage_decrease":
        return "new value = base × (1 − rate ÷ 100)"
    if t == "derivative":
        return "Differentiate the expression with respect to x."
    if t == "integral":
        return "Find the antiderivative with respect to x."
    if t == "definite_integral":
        return "Evaluate the antiderivative at the upper limit minus the lower limit."
    if t == "limit":
        return "Evaluate the limiting value as x approaches the specified point."
    if t == "factorial":
        return "n! = n × (n−1) × ... × 1"
    if t == "trigonometry":
        return "Angle is interpreted in degrees."
    if t == "finance":
        intent = result["intent"]
        if isinstance(v, dict):
            if intent == "emi":
                return "EMI uses the standard reducing-balance loan formula."
            return "Calculated using the selected interest formula."
        return "CAGR = (final ÷ initial)^(1/years) − 1"
    if t == "conversion":
        return "Applied the selected unit conversion factor."
    if t == "statistics":
        return "Computed the requested descriptive statistic from the supplied values."
    if t == "matrix":
        return f"Computed the matrix {result['operation']} exactly."
    if t == "equation":
        return "Solved the equation symbolically."
    return "Expression evaluated using the exact symbolic math engine."
