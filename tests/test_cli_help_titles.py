import pytest

from fastfuncstuff.cli_help import FfsArgumentParser, normalize_group_title


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Required Arguments", "Required arguments"),
        ("required", "Required arguments"),
        ("required inputs", "Required arguments"),
        (
            "Required Arguments (give -warp_dir OR -warp)",
            "Required arguments (give -warp_dir OR -warp)",
        ),
        ("Input / Output", "Input/output"),
        ("Input/Output", "Input/output"),
        ("inputs & scope", "Inputs & scope"),
        ("Cross-Validation Options", "Cross-validation options"),
        ("ARMA(1,1) Grid Options", "ARMA(1,1) grid options"),
        (
            "FLOBS Model Options (only used with -model FLOBS)",
            "FLOBS model options (only used with -model FLOBS)",
        ),
        ("MotSim regressors (Patriat et al. 2017)", "MotSim regressors (Patriat et al. 2017)"),
        ("STEP 2 — Response shape (the HRF).", "STEP 2 — Response shape (the HRF)."),
        ("NORDIC / phase", "NORDIC / phase"),
    ],
)
def test_group_titles_are_sentence_case(raw, want):
    assert normalize_group_title(raw) == want


def test_titles_are_normalised_on_the_parser():
    parser = FfsArgumentParser()
    group = parser.add_argument_group("Required Arguments")
    assert group.title == "Required arguments"
