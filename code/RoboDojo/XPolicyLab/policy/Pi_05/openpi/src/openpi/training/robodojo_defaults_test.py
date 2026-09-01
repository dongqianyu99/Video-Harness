from openpi.training import robodojo_defaults as defaults


def test_standard_robodojo_defaults_share_one_workspace_data_root():
    assert defaults.ROBODOJO_REPO_ID == "RoboDojo_lerobot_v30_video"
    assert defaults.ROBODOJO_DATASET_ROOT == (
        defaults.DATA_ROOT / defaults.ROBODOJO_REPO_ID
    )
    assert defaults.GUIDE_DOCUMENTS_ROOT == (
        defaults.DATA_ROOT / "video-harness" / "default" / "documents-openai"
    )
    assert defaults.GUIDE_MATERIALIZATION_CACHE_ROOT == (
        defaults.DATA_ROOT / "guide-cache" / "default"
    )
    assert (defaults.MAX_BOUNDARIES, defaults.MAX_UNITS) == (63, 62)
    assert (
        defaults.GUIDES_PER_BATCH
        * defaults.QUERIES_PER_GUIDE
        * defaults.GRADIENT_ACCUMULATION_STEPS
        == 256
    )
