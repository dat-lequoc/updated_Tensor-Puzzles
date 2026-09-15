"""Helpers for the Tensor Puzzles notebook.

The original project used torchtyping and a private Chalk fork. Both projects
are obsolete now, so this module uses public jaxtyping metadata and renders
examples directly as SVG.
"""

from __future__ import annotations

import html
import random
import re
import typing
from collections.abc import Callable, Iterable

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, integers
from IPython.display import HTML, SVG, display

tensor = torch.tensor


def _as_rows(value: object) -> list[list[object]]:
    """Convert a scalar, vector, or matrix to rows for the SVG renderer."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return [[value]]
    if not value:
        return [[]]
    if not isinstance(value[0], list):
        return [value]
    return value


def _format_value(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _matrix_svg(name: str, examples: list[dict[str, object]]) -> str:
    cell, label_h, gap = 54, 26, 12
    columns = list(examples[0]) if examples else []
    widths, heights = [], []
    for key in columns:
        matrices = [_as_rows(example[key]) for example in examples]
        widths.append(max((max((len(row) for row in matrix), default=1) for matrix in matrices), default=1))
        heights.append(sum(max(len(matrix), 1) for matrix in matrices) + gap * max(len(matrices) - 1, 0))
    total_w = sum(width * cell for width in widths) + gap * max(len(widths) - 1, 0)
    total_h = label_h + max(heights, default=1) * cell + 38
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;dominant-baseline:middle;text-anchor:middle}</style>',
        f'<text x="{total_w / 2:g}" y="16" font-size="16">{html.escape(name)}</text>',
    ]
    x0 = 0
    for key, width in zip(columns, widths):
        x = x0
        parts.append(f'<text x="{x + width * cell / 2:g}" y="{label_h - 8}">{html.escape(key)}</text>')
        y = label_h
        for example in examples:
            matrix = _as_rows(example[key])
            for row_i, row in enumerate(matrix):
                for col_i in range(width):
                    value = row[col_i] if col_i < len(row) else ""
                    text = _format_value(value)
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = 0.0
                    if isinstance(value, (bool, np.bool_)):
                        fill = "#f4a340" if value else "#f4f4f4"
                    elif numeric > 0:
                        fill = "#f4a340"
                    elif numeric < 0:
                        fill = "#6d9ee8"
                    else:
                        fill = "#f4f4f4"
                    xx, yy = x + col_i * cell, y + row_i * cell
                    parts.append(f'<rect x="{xx}" y="{yy}" width="{cell}" height="{cell}" fill="{fill}" stroke="#b7b7b7"/>')
                    parts.append(f'<text x="{xx + cell / 2:g}" y="{yy + cell / 2:g}">{html.escape(text)}</text>')
            y += max(len(matrix), 1) * cell + gap
        x0 += width * cell + gap
    parts.append("</svg>")
    return "".join(parts)


def draw_examples(name: str, examples: Iterable[dict[str, object]]) -> SVG:
    """Return a compact SVG visualization of tensor examples."""
    return SVG(_matrix_svg(name, list(examples)))


def _annotation_axes(annotation: object) -> list[str | int]:
    dim_str = getattr(annotation, "dim_str", "")
    return [int(token) if token.isdigit() else token for token in dim_str.split()]


def _dtype_for(annotation: object) -> type:
    """Choose a NumPy dtype from public jaxtyping dtype metadata."""
    dtypes = getattr(annotation, "dtypes", ())
    if isinstance(dtypes, str):
        names = {dtypes}
    elif isinstance(dtypes, (tuple, list, set)):
        names = {str(dtype) for dtype in dtypes}
    else:
        names = set()
    if names & {"bool", "bool_"}:
        return np.bool_
    if any(name.startswith(("float", "bfloat", "complex")) for name in names):
        return np.float64
    # Shaped deliberately allows any dtype; integer examples preserve the
    # original puzzle behaviour and make indexing operations straightforward.
    return np.int64


def _axis_names(axes: Iterable[str | int]) -> set[str]:
    names: set[str] = set()
    for axis in axes:
        if isinstance(axis, str):
            names.update(re.findall(r"[A-Za-z_]\w*", axis))
    return names


def _resolve_axis(axis: str | int, sizes: dict[str, int]) -> int:
    if isinstance(axis, int):
        return axis
    if axis in sizes:
        return sizes[axis]
    if not re.fullmatch(r"[A-Za-z0-9_+*/\-]+", axis):
        raise ValueError(f"Unsupported symbolic axis: {axis!r}")
    return int(eval(axis, {"__builtins__": {}}, sizes))


@composite
def spec(draw, function: Callable[..., object], min_size: int = 1):
    """Generate tensors matching a function's jaxtyping shape annotations."""
    hints = typing.get_type_hints(function, include_extras=True)
    axes_by_name = {
        name: _annotation_axes(annotation)
        for name, annotation in hints.items()
        if hasattr(annotation, "dim_str")
    }
    names = set().union(*(_axis_names(axes) for axes in axes_by_name.values()))
    sizes = {name: draw(integers(min_value=min_size, max_value=5)) for name in names}
    arrays_by_name: dict[str, np.ndarray] = {}
    for name, axes in axes_by_name.items():
        shape = tuple(_resolve_axis(axis, sizes) for axis in axes)
        dtype = _dtype_for(hints[name])
        elements = integers(min_value=-5, max_value=5) if np.issubdtype(dtype, np.integer) else None
        value = draw(arrays(shape=shape, dtype=dtype, elements=elements, unique=False))
        arrays_by_name[name] = np.nan_to_num(value, nan=0, neginf=0, posinf=0)
    arrays_by_name["return"][:] = 0
    return arrays_by_name, sizes


def make_test(
    name: str,
    problem: Callable[..., torch.Tensor],
    problem_spec: Callable[..., None],
    add_sizes: list[str] | None = None,
    constraint: Callable[[dict[str, object]], dict[str, object]] = lambda values: values,
):
    """Build a Hypothesis test for one puzzle and show a few examples."""
    add_sizes = add_sizes or []
    examples = []
    for _ in range(3):
        example, sizes = spec(problem, 3).example()
        example = constraint(example)
        out = example.pop("return").tolist()
        problem_spec(*example.values(), out)
        for size in add_sizes:
            example[size] = sizes[size]
        try:
            example["yours"] = problem(*map(tensor, example.values()))
        except NotImplementedError:
            pass
        for size in add_sizes:
            example.pop(size, None)
        example["target"] = tensor(out)
        examples.append(example)
    display(draw_examples(name, examples))

    @settings(deadline=None)
    @given(spec(problem))
    def test_problem(generated):
        values, sizes = generated
        values = constraint(values)
        out = values.pop("return").tolist()
        problem_spec(*values.values(), out)
        for size in add_sizes:
            values[size] = sizes[size]
        actual = problem(*map(tensor, values.values()))
        expected = tensor(out)
        actual = torch.broadcast_to(actual, expected.shape)
        torch.testing.assert_close(expected, actual)

    return test_problem


def run_test(test: Callable[[], None]) -> HTML:
    """Run a puzzle test and return the traditional celebratory puppy video."""
    test()
    print("Correct!")
    pups = [
        "2m78jPG", "pn1e9TO", "MQCIwzT", "udLK6FS", "ZNem5o3", "DS2IZ6f", "aydRUz8",
        "MVUdQYK", "kLvno0p", "wScLiVz", "Z0TII8i", "F1SChho", "9hRi2jN", "lvzRF3W",
    ]
    return HTML(f'<video alt="test" controls autoplay><source src="https://openpuppies.com/mp4/{random.choice(pups)}.mp4" type="video/mp4"></video>')
