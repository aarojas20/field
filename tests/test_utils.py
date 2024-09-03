import sys
sys.path.append("../src")
import utils


def test_determine_zone():
    assert utils.determine_zone(-71.17) == 19 # Arlington, MA
    assert utils.determine_zone(-122.265) == 10 # Berkeley, CA
    assert utils.determine_zone(-3.735) == 30 # Madrid, Spain
    assert utils.determine_zone(151.150) == 56 # Sydney, AU


def test_determine_hemisphere():
    assert utils.determine_hemisphere(40) == "north"
    assert utils.determine_hemisphere(-33) == "south"


def test_determine_crs_code():
    assert utils.determine_crs_code(-71.17, 40) == "32619"
    assert utils.determine_crs_code(151.150, -33) == "32756"


def test_convert_degree_to_meters():
    pass
