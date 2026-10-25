"""Encoding invariance of the fixpoint normaliser (carta/normalize.py)."""
from urllib.parse import quote

import pytest

from carta.evasion import form_encode
from carta.normalize import canonical_key, normalize, normalize_single_pass, twin_key

VALUES = [
    "1' OR '1'='1",
    "<script>alert(1)</script>",
    "../../etc/passwd",
    "admin'--",
    "café crème",
    "a+b c&d=e",
    "x; cat /etc/passwd",
    "union select username, password from users",
]


@pytest.mark.parametrize("v", VALUES)
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_extra_encoding_layers_do_not_change_the_normalised_text(v, k):
    e = v
    for _ in range(k):
        e = quote(e, safe="")
    assert normalize(form_encode(e)) == normalize(form_encode(v))


def test_single_pass_normaliser_is_not_invariant():
    v = "1' OR '1'='1"
    once = quote(v, safe="")
    assert normalize_single_pass(form_encode(once)) != normalize_single_pass(form_encode(v))


def test_inline_comments_whitespace_and_case_are_folded():
    assert normalize("UNION/**/SELECT") == normalize("union  select")


def test_zero_width_characters_are_removed():
    assert normalize("se\u200blect") == "select"
    assert normalize("a\ufeffb\u200dc\u200cd") == "abcd"


def test_canonical_key_ignores_encoding_but_twin_key_ignores_method():
    assert canonical_key("GET", "/a", "id=%2531", "") == canonical_key("GET", "/a", "id=1", "")
    assert canonical_key("GET", "/a", "id=1", "") != canonical_key("POST", "/a", "", "id=1")
    assert twin_key("GET", "/a", "id=1&x=2", "") == twin_key("POST", "/a", "", "x=2&id=1")
