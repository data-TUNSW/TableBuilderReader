import pandas as pd
import pytest

from abs_methods import geographic_weightings as gw


@pytest.fixture
def lga_shapefile_frame():
    """Stands in for LGA_2025_AUST_GDA2020.shp, with a padded name."""
    return pd.DataFrame(
        {
            "LGA_CODE25": ["17200", "11300"],
            "LGA_NAME25": ["Sydney ", " Inner West"],
            "STE_CODE21": ["1", "1"],
            "STE_NAME21": ["New South Wales", "New South Wales"],
            "AREASQKM": [26.6762, 35.1],
            "AUS_CODE21": ["AUS", "AUS"],
            "LOCI_URI21": ["http://x", "http://y"],
            "geometry": [None, None],
        }
    )


def test_boilerplate_and_geometry_columns_are_dropped(lga_shapefile_frame):
    attrs = gw.clean_shapefile_attributes(lga_shapefile_frame, "LGA_NAME25")
    assert "AUS_CODE21" not in attrs.columns
    assert "LOCI_URI21" not in attrs.columns
    assert "geometry" not in attrs.columns


def test_area_column_is_prefixed_with_the_name_column(lga_shapefile_frame):
    attrs = gw.clean_shapefile_attributes(lga_shapefile_frame, "LGA_NAME25")
    assert "LGA_NAME25_AREASQKM" in attrs.columns
    assert "AREASQKM" not in attrs.columns


def test_merge_key_is_whitespace_stripped(lga_shapefile_frame):
    attrs = gw.clean_shapefile_attributes(lga_shapefile_frame, "LGA_NAME25")
    assert attrs["LGA_NAME25"].tolist() == ["Sydney", "Inner West"]


def test_missing_name_column_raises(lga_shapefile_frame):
    with pytest.raises(ValueError, match="LGA_NAME99"):
        gw.clean_shapefile_attributes(lga_shapefile_frame, "LGA_NAME99")


def test_attach_strips_the_left_key_too_so_padded_names_still_match(
    lga_shapefile_frame,
):
    """Review Focus 5: an unstripped key makes every attribute silently null."""
    attrs = gw.clean_shapefile_attributes(lga_shapefile_frame, "LGA_NAME25")
    df = pd.DataFrame({"Local Government Area (2025)": ["Sydney  ", "Inner West"]})

    merged = gw.attach_shapefile_attributes(
        df,
        attrs,
        on="Local Government Area (2025)",
        name_column="LGA_NAME25",
    )

    assert merged["LGA_CODE25"].tolist() == ["17200", "11300"]
    assert merged["LGA_CODE25"].isna().sum() == 0


def test_attach_drops_the_duplicated_name_column(lga_shapefile_frame):
    attrs = gw.clean_shapefile_attributes(lga_shapefile_frame, "LGA_NAME25")
    df = pd.DataFrame({"Local Government Area (2025)": ["Sydney"]})

    merged = gw.attach_shapefile_attributes(
        df,
        attrs,
        on="Local Government Area (2025)",
        name_column="LGA_NAME25",
    )

    assert "LGA_NAME25" not in merged.columns
