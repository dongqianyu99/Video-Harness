import ast
from pathlib import Path

SOURCE = Path(__file__).with_name("smoke_guided_forward_backward.py").read_text()


def _gradient_sharding_helper():
    tree = ast.parse(SOURCE)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_gradient_sharding")
    namespace = {}
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, "<helper>", "exec"), namespace)
    return namespace["_gradient_sharding"]


def test_smoke_scopes_query_episode_to_guided_data_config():
    assert "query_episode_indices=(args.query_episode_index,)" in SOURCE


def test_smoke_uses_mesh_context_and_sharded_jit_for_both_modes():
    assert "with mesh_module.set_mesh(mesh):" in SOURCE
    assert "in_shardings=(replicated_sharding, train_state_sharding, batch_sharding)" in SOURCE
    assert "out_shardings=(train_state_sharding, replicated_sharding)" in SOURCE
    assert "out_shardings=(replicated_sharding, grad_sharding, replicated_sharding, replicated_sharding)" in SOURCE


def test_smoke_has_one_execution_path_per_mode():
    assert "if args.no_optimizer_update:" in SOURCE
    assert SOURCE.count("guided_loss_and_grad(") == 1
    assert SOURCE.count("guided_train_step(") == 1
    assert "updated_loss" not in SOURCE


def test_gradient_sharding_filters_trainable_params_only():
    class FakeParamsSharding:
        def filter(self, predicate):
            self.predicate = predicate
            return ("trainable-only", predicate)

    class FakeStateSharding:
        params = FakeParamsSharding()

    predicate = object()
    result = _gradient_sharding_helper()(FakeStateSharding(), predicate)
    assert result == ("trainable-only", predicate)
    assert FakeStateSharding.params.predicate is predicate
