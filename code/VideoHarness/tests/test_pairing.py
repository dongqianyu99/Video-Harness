from video_harness.pairing import build_pairs
from video_harness.robodojo import EpisodeRecord
from video_harness.sampling import BEHAVIOR_DOCUMENT_SCHEMA_VERSION


def _record(episode: int, task: int) -> EpisodeRecord:
    return EpisodeRecord(
        episode_index=episode,
        task_index=task,
        task_instruction=f"task-{task}",
        task_kind="benchmark",
        length=10,
        dataset_from_index=episode * 10,
        dataset_to_index=(episode + 1) * 10,
        data_path="data.parquet",
        videos=(),
    )


def test_pairs_are_same_task_and_episode_disjoint() -> None:
    records = [
        _record(episode, task)
        for task in range(2)
        for episode in range(task * 3, task * 3 + 3)
    ]
    pairs = build_pairs(records, build_id="test-build", supports_per_query=2, seed=4)
    by_episode = {record.episode_index: record for record in records}
    assert len(pairs) == 12
    for pair in pairs:
        query = by_episode[pair["query_episode_index"]]
        support = by_episode[pair["support_episode_index"]]
        assert query.task_index == support.task_index
        assert query.episode_index != support.episode_index
        assert pair["support_document_id"].endswith(f"{support.episode_index:07d}")
        assert pair["guide_schema_version"] == BEHAVIOR_DOCUMENT_SCHEMA_VERSION
        assert pair["build_id"] == "test-build"


def test_pairing_is_deterministic() -> None:
    records = [_record(episode, 0) for episode in range(5)]
    assert build_pairs(records, build_id="test-build", seed=9) == build_pairs(
        reversed(records), build_id="test-build", seed=9
    )
