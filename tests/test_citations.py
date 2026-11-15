from app.services.citations import find_unsupported_citations, strip_unsupported_citations

REF = (
    "[1] Attention Is All You Need (Vaswani, 2017)\nWe propose the Transformer.\n"
    "[2] Deep Residual Learning (He, 2016)\nResidual connections ease training.\n"
)


def test_grounded_author_year_is_kept():
    text = "Transformers replaced recurrence (Vaswani, 2017) and residuals help [He et al., 2016]."
    assert find_unsupported_citations(text, REF) == []


def test_fabricated_author_is_flagged():
    text = "This was shown earlier (Smith, 2019)."
    assert find_unsupported_citations(text, REF) == ["Smith, 2019"]


def test_right_author_wrong_year_is_flagged():
    assert find_unsupported_citations("(Vaswani, 2020)", REF) == ["Vaswani, 2020"]


def test_multi_citation_group_keeps_supported_parts():
    cleaned, removed = strip_unsupported_citations("Claim (Vaswani, 2017; Smith, 2019).", REF)
    assert removed == ["Smith, 2019"]
    assert cleaned == "Claim (Vaswani, 2017)."


def test_fully_unsupported_group_is_removed_and_spacing_tidied():
    cleaned, removed = strip_unsupported_citations("A claim (Smith, 2019). Next.", REF)
    assert removed == ["Smith, 2019"]
    assert cleaned == "A claim. Next."


def test_no_reference_text_means_author_year_not_judged():
    assert find_unsupported_citations("(Smith, 2019)", "") == []


def test_numbered_refs_and_non_citations_untouched():
    text = "See [1] and (Figure 2) and (n = 120) and [Section 3.1]."
    assert find_unsupported_citations(text, REF) == []


def test_source_style_checked_against_known_sources():
    text = "From the draft (Source: thesis.docx) and (Source: made_up.pdf)."
    assert find_unsupported_citations(text, known_sources={"thesis.docx"}) == ["(Source: made_up.pdf)"]


def test_source_style_not_judged_without_known_sources():
    assert find_unsupported_citations("(Source: anything.pdf)") == []


def test_et_al_and_pages():
    assert find_unsupported_citations("(Vaswani et al., 2017, p. 4)", REF) == []
