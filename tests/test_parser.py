import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.parser import extract_with_pymupdf, parse_tree, _sec_level

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
V1 = os.path.join(DATA, "ct200_manual.pdf")
V2 = os.path.join(DATA, "ct200_manual_v2.pdf")


def parse_v1():
    return parse_tree(extract_with_pymupdf(V1))

def parse_v2():
    return parse_tree(extract_with_pymupdf(V2))


def test_sec_level():
    assert _sec_level("1") == 1
    assert _sec_level("2.1") == 2
    assert _sec_level("2.1.1.1") == 4


def test_v1_node_count():
    nodes, _ = parse_v1()
    secs = [n for n in nodes if n.section_number]
    # 1, 1.1, 1.2, 2, 2.1, 2.1.1.1, 2.2, 3, 3.1, 3.2, 3.3, 3.4,
    # 4, 4.1, 4.2, 4.3, 5, 5.1, 5.2, 6, 6.1, 6.2, 7, 7.1, 7.2, 8, 8.1 = 27
    assert len(secs) == 27, f"got {len(secs)}"


def test_v2_has_extra_section():
    nodes, _ = parse_v2()
    secs = [n for n in nodes if n.section_number]
    assert len(secs) == 28  # v2 adds 5.3


# edge case 1: duplicate headings
# "Error Codes" is both 4.2 and 7.1. they need to be separate nodes.

def test_duplicate_headings_two_nodes():
    nodes, _ = parse_v1()
    ec = [n for n in nodes if n.heading.strip() == "Error Codes"]
    assert len(ec) == 2

def test_duplicate_headings_different_ids():
    nodes, _ = parse_v1()
    ec = [n for n in nodes if n.heading.strip() == "Error Codes"]
    ids = {n.logical_id for n in ec}
    assert len(ids) == 2  # sec_4.2 and sec_7.1

def test_duplicate_headings_correct_parents():
    nodes, _ = parse_v1()
    ec = [n for n in nodes if n.heading.strip() == "Error Codes"]
    s42 = next(n for n in ec if n.section_number == "4.2")
    s71 = next(n for n in ec if n.section_number == "7.1")
    assert s42.parent.section_number == "4"
    assert s71.parent.section_number == "7"


# edge case 2: out-of-order sections
# 3.4 appears before 3.3 in the pdf

def test_out_of_order_both_exist():
    nodes, _ = parse_v1()
    assert any(n.section_number == "3.3" for n in nodes)
    assert any(n.section_number == "3.4" for n in nodes)

def test_out_of_order_correct_parent():
    nodes, _ = parse_v1()
    s33 = next(n for n in nodes if n.section_number == "3.3")
    s34 = next(n for n in nodes if n.section_number == "3.4")
    assert s33.parent.section_number == "3"
    assert s34.parent.section_number == "3"

def test_out_of_order_detected():
    _, edge_cases = parse_v1()
    assert any("Out-of-order" in e for e in edge_cases)

def test_out_of_order_positions():
    # 3.4 shows up first in the doc so it gets a lower position
    nodes, _ = parse_v1()
    s33 = next(n for n in nodes if n.section_number == "3.3")
    s34 = next(n for n in nodes if n.section_number == "3.4")
    assert s34.position < s33.position


# edge case 3: level skipping
# 2.1.1.1 exists but 2.1.1 doesn't

def test_level_skip_exists():
    nodes, _ = parse_v1()
    assert any(n.section_number == "2.1.1.1" for n in nodes)

def test_level_skip_parent_is_2_1():
    nodes, _ = parse_v1()
    n = next(n for n in nodes if n.section_number == "2.1.1.1")
    assert n.parent.section_number == "2.1"

def test_level_skip_detected():
    _, ec = parse_v1()
    assert any("Level skip" in e for e in ec)

def test_level_skip_level_is_4():
    nodes, _ = parse_v1()
    n = next(n for n in nodes if n.section_number == "2.1.1.1")
    assert n.level == 4



def test_same_parse_same_hash():
    n1, _ = parse_v1()
    n2, _ = parse_v1()
    s1 = next(n for n in n1 if n.section_number == "1")
    s2 = next(n for n in n2 if n.section_number == "1")
    assert s1.content_hash == s2.content_hash

def test_changed_section_different_hash():
    v1, _ = parse_v1()
    v2, _ = parse_v2()
    # 2.1.1.1 changed: 300->250 cycles, 15%->10% threshold
    b1 = next(n for n in v1 if n.section_number == "2.1.1.1")
    b2 = next(n for n in v2 if n.section_number == "2.1.1.1")
    assert b1.content_hash != b2.content_hash

def test_unchanged_section_same_hash():
    v1, _ = parse_v1()
    v2, _ = parse_v2()
    # 1.1 didn't change
    s1 = next(n for n in v1 if n.section_number == "1.1")
    s2 = next(n for n in v2 if n.section_number == "1.1")
    assert s1.content_hash == s2.content_hash



def test_v2_has_5_3():
    nodes, _ = parse_v2()
    assert any(n.section_number == "5.3" for n in nodes)

def test_v1_no_5_3():
    nodes, _ = parse_v1()
    assert not any(n.section_number == "5.3" for n in nodes)


def test_table_flag_2_1():
    nodes, _ = parse_v1()
    n = next(n for n in nodes if n.section_number == "2.1")
    assert n.has_table

def test_table_flag_4_2():
    nodes, _ = parse_v1()
    n = next(n for n in nodes if n.section_number == "4.2")
    assert n.has_table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
