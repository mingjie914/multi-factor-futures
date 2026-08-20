from __future__ import annotations

import pandas as pd

from data.contract_symbols import contract_symbol_parts


def test_contract_roots_are_parsed_exactly_for_one_and_two_letter_roots():
    symbols = pd.Series(
        [
            "T2609",
            "TL2609",
            "A2609",
            "AD2609",
            "AO2609",
            "AP2609",
            "P2609",
            "PB2609",
            "PF2609",
            "PG2609",
            "PP2609",
            "PS2609",
            "PX2609",
        ]
    )

    parsed = contract_symbol_parts(symbols)

    assert parsed["is_concrete"].all()
    assert parsed.loc[0, "root"] == "T"
    assert parsed.loc[1, "root"] == "TL"
    assert parsed.loc[2:5, "root"].tolist() == ["A", "AD", "AO", "AP"]
    assert parsed.loc[6:, "root"].tolist() == [
        "P",
        "PB",
        "PF",
        "PG",
        "PP",
        "PS",
        "PX",
    ]


def test_only_root_plus_four_digit_valid_month_is_concrete():
    symbols = pd.Series(
        [
            "T2601",
            "TL2612",
            "A2609",
            "A",
            "T",
            "A00",
            "A01",
            "A8888",
            "A9999",
            "8888",
            "9999",
            "AG02",
            "AG10M",
            "A2600",
            "A2613",
            "A260",
            "A26013",
        ]
    )

    parsed = contract_symbol_parts(symbols)

    assert parsed.loc[:2, "is_concrete"].tolist() == [True, True, True]
    assert not parsed.loc[3:, "is_concrete"].any()
    assert parsed.loc[0, "delivery_month"] == 1
    assert parsed.loc[1, "delivery_month"] == 12
