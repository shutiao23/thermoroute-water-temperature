from pathlib import Path

from thermoroute.spatial import huc2_cluster_map, load_station_registry


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_registry_preserves_stable_ids_and_leading_zero_huc() -> None:
    registry = load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    assert len(registry) == 120
    assert registry.site_no.nunique() == 120
    assert registry.site_no.str.fullmatch(r"\d{8,15}").all()
    assert registry.huc_cd.str.fullmatch(r"(?:\d{8}|\d{12})").all()
    assert registry.huc2.str.fullmatch(r"\d{2}").all()
    assert (registry.huc_cd.str[:2] == registry.huc2).all()
    clusters = huc2_cluster_map(registry)
    assert set(clusters) == set(registry.site_no)
    assert all(value.startswith("HUC2:") for value in clusters.values())
