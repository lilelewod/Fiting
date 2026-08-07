import pytest

from tools.record_pdf_visual_qa import parse_page_spec


def test_parse_page_spec_expands_ranges_and_deduplicates_pages():
    assert parse_page_spec("1-3, 3, 5") == [1, 2, 3, 5]


@pytest.mark.parametrize("spec", ["", "0", "3-2", "-1"])
def test_parse_page_spec_rejects_empty_or_invalid_confirmation(spec):
    with pytest.raises((ValueError, TypeError)):
        parse_page_spec(spec)
