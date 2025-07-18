"""test_level3_dithers: Test of spectrographic rules."""

import re

import pytest
from astropy.utils.data import get_pkg_data_filename

from jwst.associations import generate
from jwst.associations.lib.utilities import constrain_on_candidates
from jwst.associations.tests.helpers import (
    BasePoolRule,
    PoolParams,
    combine_pools,
    registry_level3_only,
)


class TestLevel3Spec(BasePoolRule):
    pools = [
        PoolParams(
            path=get_pkg_data_filename(
                "data/pool_005_spec_niriss.csv", package="jwst.associations.tests"
            ),
            n_asns=0,
            n_orphaned=0,
        ),
        PoolParams(
            path=get_pkg_data_filename(
                "data/pool_006_spec_nirspec.csv", package="jwst.associations.tests"
            ),
            n_asns=3,
            n_orphaned=0,
        ),
        PoolParams(
            path=get_pkg_data_filename(
                "data/pool_007_spec_miri.csv", package="jwst.associations.tests"
            ),
            n_asns=1,
            n_orphaned=0,
        ),
        PoolParams(
            path=get_pkg_data_filename(
                "data/pool_019_niriss_wfss.csv", package="jwst.associations.tests"
            ),
            n_asns=3,
            n_orphaned=0,
        ),
    ]

    valid_rules = [
        "Asn_Lv3Image",
    ]


def nirspec_params_id(fixture_value):
    """Give fixture parameters ids"""
    cid, asn_type, asn_name, product_name, exptypes = fixture_value
    return asn_name


@pytest.fixture(
    scope="module",
    params=[
        (
            "o001",
            "spec3",
            r"jw99009-o001_spec3_\d{5}_asn",
            "jw99009-o001_t001-{source_id}_nirspec_f100lp-g140m-{slit_name}-s200a2",
            set(("science", "target_acquisition", "autowave")),
        ),
        (
            "o002",
            "spec3",
            r"jw99009-o002_spec3_\d{5}_asn",
            "jw99009-o002_{source_id}_nirspec_f100lp-g140h",
            set(("science", "target_acquisition", "autoflat", "autowave")),
        ),
        (
            "o003",
            "spec3",
            r"jw99009-o003_spec3_\d{5}_asn",
            "jw99009-o003_t002_nirspec_g235h",
            set(("science", "target_acquisition", "autowave")),
        ),
    ],
    ids=nirspec_params_id,
)
def nirspec_params(request):
    cid, asn_type, asn_name, product_name, exptypes = request.param
    pool = combine_pools(
        get_pkg_data_filename("data/pool_006_spec_nirspec.csv", package="jwst.associations.tests")
    )
    gc = constrain_on_candidates((cid,))
    rules = registry_level3_only(global_constraints=gc)
    asns = generate(pool, rules)
    return asns, asn_type, asn_name, product_name, exptypes


def test_nirspec_modes(nirspec_params):
    asns, asn_type, asn_name, product_name, exptypes = nirspec_params

    assert len(asns) == 1
    asn = asns[0]
    assert asn["asn_type"] == asn_type
    assert re.match(asn_name, asn.asn_name)
    product = asn["products"][0]
    assert product["name"] == product_name
    found_exptypes = set(member["exptype"] for member in product["members"])
    assert found_exptypes == exptypes


@pytest.fixture(
    scope="module",
    params=[
        (
            "o007",
            "spec3",
            r"jw99009-o007_spec3_\d{5}_asn",
            "jw99009-o007_t001_miri",
        ),
        (
            "o008",
            "spec3",
            r"jw99009-o008_spec3_\d{5}_asn",
            "jw99009-o008_t001_miri",
        ),
    ],
)
def miri_params(request):
    cid, asn_type, asn_name, product_name = request.param
    pool = combine_pools(
        get_pkg_data_filename("data/pool_007_spec_miri.csv", package="jwst.associations.tests")
    )
    gc = constrain_on_candidates((cid,))
    rules = registry_level3_only(global_constraints=gc)
    asns = generate(pool, rules)
    return asns, asn_type, asn_name, product_name


def test_miri_modes(miri_params):
    asns, asn_type, asn_name, product_name = miri_params

    assert len(asns) == 1
    asn = asns[0]
    assert asn["asn_type"] == asn_type
    assert re.match(asn_name, asn.asn_name)
    assert asn["products"][0]["name"] == product_name
