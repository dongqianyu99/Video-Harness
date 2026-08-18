from __future__ import annotations

import pytest

from XPolicyLab.policy.Pi_05 import deploy


class _Client:
    def __init__(self, *, fail_update=False):
        self.calls = []
        self.fail_update = fail_update

    def call(self, func_name=None, obs=None):
        self.calls.append((func_name, obs))
        if self.fail_update and func_name == "update_obs":
            raise RuntimeError("update failed")
        if func_name == "get_action":
            return [[0.0], [1.0]]
        return None


class _Env:
    task_name = "make_toast"

    def __init__(self):
        self.ended = False
        self.actions = []

    def is_episode_end(self):
        return self.ended

    def get_obs(self):
        return {"instruction": "toast bread"}

    def take_action(self, action):
        self.actions.append(action)
        self.ended = True


def test_eval_lifecycle_prepares_before_reset_and_always_ends_trial():
    env = _Env()
    client = _Client()

    deploy.eval_one_episode(env, client)

    names = [name for name, _ in client.calls]
    assert names[:4] == ["prepare_case", "reset", "update_obs", "get_action"]
    assert names[-1] == "trial_end"
    assert client.calls[0][1] == {"task_name": "make_toast"}
    assert env.actions == [[0.0]]


def test_trial_end_runs_when_episode_loop_raises():
    env = _Env()
    client = _Client(fail_update=True)

    with pytest.raises(RuntimeError, match="update failed"):
        deploy.eval_one_episode(env, client)

    assert client.calls[-1] == ("trial_end", {"episode_end": True})
