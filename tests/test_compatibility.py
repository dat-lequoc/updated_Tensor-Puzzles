import torch
from jaxtyping import Shaped

from lib import draw_examples, make_test, spec


def test_latest_torch_and_svg_helpers():
    assert torch.__version__.split("+")[0] == "2.14.0"
    rendered = draw_examples("demo", [{"x": torch.tensor([1, -1]), "ret": torch.tensor([0, 2])}])
    assert rendered.data.startswith("<svg")
    assert "demo" in rendered.data


def test_jaxtyping_spec_and_hypothesis_harness():
    def ones(i: int) -> Shaped[torch.Tensor, "i"]:  # noqa: F821
        return torch.ones(i, dtype=torch.int64)

    def ones_spec(out):
        for index in range(len(out)):
            out[index] = 1

    generated, sizes = spec(ones, 3).example()
    assert generated["return"].shape == (sizes["i"],)
    test_ones = make_test("ones", ones, ones_spec, add_sizes=["i"])
    test_ones()
