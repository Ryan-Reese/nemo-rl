# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.dpo import (
    DPOConfig,
    MasterConfig,
    _get_dpo_save_state,
    _initial_dpo_save_state,
    add_ref_logprobs_to_data,
    dpo_train,
    setup,
)
from nemo_rl.algorithms.loss import DPOLossConfig, DPOLossFn, PreferenceLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.logger import LoggerConfig


def test_get_dpo_save_state_handles_legacy_checkpoint_and_filters_metrics():
    assert _get_dpo_save_state({}) == _initial_dpo_save_state()

    loaded_state = {
        "epoch": 1,
        "step": 3,
        "total_steps": 13,
        "consumed_samples": 32,
        "val:accuracy": 0.75,
    }

    save_state = _get_dpo_save_state(loaded_state)

    assert vars(save_state) == {
        "epoch": 1,
        "step": 3,
        "total_steps": 13,
        "consumed_samples": 32,
        "total_valid_tokens": 0,
    }
    assert "total_valid_tokens" not in loaded_state


def test_get_dpo_save_state_restores_mpo_reward_shift():
    from nemo_rl.algorithms.mpo import MPOSaveState

    def initial_state() -> MPOSaveState:
        return MPOSaveState(
            epoch=0,
            step=0,
            total_steps=0,
            consumed_samples=0,
            total_valid_tokens=0,
            reward_shift=0.25,
            reward_shift_num_updates=0,
        )

    save_state = _get_dpo_save_state(
        {
            "total_steps": 12,
            "reward_shift": -1.5,
            "reward_shift_num_updates": 12,
            "val:accuracy": 0.75,
        },
        save_state_cls=MPOSaveState,
        initial_save_state_fn=initial_state,
    )

    assert save_state == MPOSaveState(
        epoch=0,
        step=0,
        total_steps=12,
        consumed_samples=0,
        total_valid_tokens=0,
        reward_shift=-1.5,
        reward_shift_num_updates=12,
    )


class MockPolicy:
    def __init__(self, logprobs):
        self.logprobs = logprobs
        self.sharding_annotations = NamedSharding(
            layout=np.arange(2).reshape(
                1,  # PP
                -1,  # DP
                1,  # CP
                1,  # TP
            ),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )

    def get_reference_policy_logprobs(self, batch, micro_batch_size):
        return {"reference_logprobs": self.logprobs}


@pytest.mark.parametrize(
    ("num_pairs", "is_val", "expected_rows"),
    [(1, False, 8), (2, False, 8), (4, False, 8), (1, True, 4), (3, True, 8)],
)
def test_add_logprobs_to_batch(
    num_pairs: int, is_val: bool, expected_rows: int
) -> None:
    """Test that add_ref_logprobs_to_data correctly adds reference policy logprobs to batches."""
    # Create mock data
    batch_size = num_pairs * 2
    seq_len = 4
    vocab_size = 16

    # Create a mock batch
    mock_batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "sample_mask": torch.ones(batch_size),
        "pair_index": torch.arange(num_pairs).repeat_interleave(2),
        "is_chosen": torch.arange(batch_size) % 2 == 0,
    }

    # Create mock logprobs that will be returned by the policy
    mock_logprobs = torch.randn(expected_rows, seq_len)

    # Create a mock dataloader that yields our mock batch
    mock_dataloader = MagicMock()
    mock_dataloader.__iter__.return_value = iter([BatchedDataDict(mock_batch)])

    # Create a mock policy that returns our mock logprobs
    mock_policy = MockPolicy(mock_logprobs)

    # Create a mock master config
    mock_master_config = MasterConfig.model_construct(
        policy={"train_micro_batch_size": 1, "train_global_batch_size": 4},
        dpo=DPOConfig(val_micro_batch_size=1),
    )

    # Get the augmented batches
    augmented_batches = list(
        add_ref_logprobs_to_data(
            mock_dataloader, mock_policy, mock_master_config, is_val=is_val
        )
    )

    # Verify we got exactly one batch
    assert len(augmented_batches) == 1
    augmented_batch = augmented_batches[0]

    # Verify the original batch data is preserved
    assert torch.equal(
        augmented_batch["input_ids"][:batch_size], mock_batch["input_ids"]
    )
    assert torch.equal(
        augmented_batch["attention_mask"][:batch_size], mock_batch["attention_mask"]
    )
    assert augmented_batch.size == expected_rows
    assert augmented_batch["sample_mask"].sum().item() == batch_size
    assert not augmented_batch["sample_mask"][batch_size:].any()
    assert torch.equal(
        augmented_batch["pair_index"][::2], augmented_batch["pair_index"][1::2]
    )
    assert augmented_batch["pair_index"][::2].unique().numel() == expected_rows // 2
    assert augmented_batch["is_chosen"].tolist() == [True, False] * (expected_rows // 2)

    # Verify the reference policy logprobs were added correctly
    assert "reference_policy_logprobs" in augmented_batch
    assert augmented_batch["reference_policy_logprobs"].shape == (
        expected_rows,
        seq_len,
    )

    # Verify the logprobs were rolled by -1 as expected
    expected_logprobs = torch.roll(mock_logprobs, -1, dims=-1)
    assert torch.equal(augmented_batch["reference_policy_logprobs"], expected_logprobs)


@pytest.fixture
def mock_dpo_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {
            "loss": [0.5],
            "sft_loss": [0.3],
            "preference_loss": [0.2],
            "accuracy": [1.0],
            "rewards_chosen_mean": [4.5],
            "rewards_rejected_mean": [3.5],
            "num_valid_samples": [1.0],
            "global_valid_seqs": [1.0],
            "global_valid_toks": [10],
        },
    }
    policy.get_reference_policy_logprobs.side_effect = lambda batch, **_: {
        "reference_logprobs": torch.zeros_like(batch["input_ids"], dtype=torch.float32)
    }
    policy.sharding_annotations = NamedSharding(
        layout=np.arange(1).reshape(1, -1, 1, 1),  # 1 GPU to match cluster config
        names=[
            "pipeline_parallel",
            "data_parallel",
            "context_parallel",
            "tensor_parallel",
        ],
    )

    # Match the collated rows for the configured two-pair global batch.
    mock_batch = BatchedDataDict(
        {
            "input_ids": torch.ones(4, 10, dtype=torch.long),
            "input_lengths": torch.full((4,), 10),
            "token_mask": torch.ones(4, 10),
            "sample_mask": torch.ones(4),
            "pair_index": torch.tensor([0, 0, 1, 1]),
            "is_chosen": torch.tensor([True, False, True, False]),
        }
    )

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = PreferenceLossFn()
    logger = MagicMock()
    checkpointer = MagicMock()

    # Create mock master config
    master_config = MasterConfig.model_construct(
        **{
            "dpo": DPOConfig.model_construct(
                max_num_steps=5,
                max_num_epochs=2,
                val_period=100,
                val_batches=1,
                val_global_batch_size=1,
                val_micro_batch_size=1,
                val_at_start=False,
                val_at_end=False,
            ),
            "policy": {
                "train_global_batch_size": 2,
                "make_sequence_length_divisible_by": 1,
                "reward_model_cfg": {
                    "enabled": True,
                    "reward_model_type": "bradley_terry",
                },
                "train_micro_batch_size": 1,
            },
            "checkpointing": CheckpointingConfig.model_construct(
                enabled=False, checkpoint_must_save_by=None, save_period=10
            ),
            "cluster": ClusterConfig(num_nodes=1, gpus_per_node=1),
        }
    )

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "master_config": master_config,
    }


@pytest.mark.parametrize(
    ("num_pairs", "sample_weight"),
    [(1, 1.0), (4, 1.0), (16, 1.0), (260, 1.0), (1, 0.5)],
)
def test_training_retains_all_pairs_each_epoch(
    mock_dpo_components: dict, num_pairs: int, sample_weight: float, tmp_path: Path
) -> None:
    components = mock_dpo_components
    cfg = components["master_config"]
    cfg.policy.update(
        train_global_batch_size=16,
        dynamic_batching={"enabled": False},
        sequence_packing={"enabled": False},
        megatron_cfg={"enabled": False},
    )
    cfg.dpo.max_num_steps = 100
    cfg.dpo.val_period = 0
    cfg.data = {"shuffle": True, "num_workers": 0}
    cfg.logger = LoggerConfig(log_dir=str(tmp_path))
    cfg.cluster = ClusterConfig(num_nodes=1, gpus_per_node=2)
    policy = components["policy"]
    policy.sharding_annotations = MockPolicy(None).sharding_annotations
    checkpointer = components["checkpointer"]
    checkpointer.get_latest_checkpoint_path.return_value = None
    checkpointer.load_training_info.return_value = None
    checkpointer.get_resume_paths.return_value = (None, None)
    dataset = [
        {
            "message_log_chosen": [
                {"role": "user", "token_ids": torch.tensor([index + 1])},
                {"role": "assistant", "token_ids": torch.tensor([2, 3])},
            ],
            "message_log_rejected": [
                {"role": "user", "token_ids": torch.tensor([index + 1])},
                {"role": "assistant", "token_ids": torch.tensor([4, 5])},
            ],
            "length_chosen": 3,
            "length_rejected": 3,
            "loss_multiplier": sample_weight,
            "idx": index,
        }
        for index in range(num_pairs)
    ]

    def reference_logprobs(batch: BatchedDataDict, **kwargs: object) -> dict:
        assert batch.size == 32
        return {
            "reference_logprobs": torch.zeros_like(
                batch["input_ids"], dtype=torch.float32
            )
        }

    policy.get_reference_policy_logprobs.side_effect = reference_logprobs
    with (
        patch("nemo_rl.algorithms.dpo.Logger", return_value=components["logger"]),
        patch("nemo_rl.algorithms.dpo.CheckpointManager", return_value=checkpointer),
        patch("nemo_rl.algorithms.dpo.RayVirtualCluster"),
        patch("nemo_rl.algorithms.dpo.Policy", return_value=policy),
    ):
        _, _, train_loader, val_loaders, loss_fn, _, _, state, _ = setup(
            cfg, components["tokenizer"], dataset, {}
        )
    updates_per_epoch = math.ceil(num_pairs / 16)
    assert len(train_loader) == updates_per_epoch
    dpo_train(
        policy,
        train_loader,
        val_loaders,
        components["tokenizer"],
        loss_fn,
        cfg,
        components["logger"],
        checkpointer,
        state,
    )
    assert policy.train.call_count == updates_per_epoch * 2
    assert state.consumed_samples == num_pairs * 2
    for epoch in range(2):
        seen = []
        for call in policy.train.call_args_list[
            epoch * updates_per_epoch : (epoch + 1) * updates_per_epoch
        ]:
            batch = call.args[0]
            real_chosen = batch["sample_mask"].bool() & batch["is_chosen"]
            seen.extend(batch["input_ids"][real_chosen, 0].tolist())
        assert sorted(seen) == list(range(1, num_pairs + 1))


@pytest.mark.parametrize("sft_loss_weight", [0.0, 0.25])
def test_masked_tail_does_not_change_loss_metrics_or_gradients(
    sft_loss_weight: float,
) -> None:
    batch = BatchedDataDict(
        input_ids=torch.ones(6, 4, dtype=torch.long),
        token_mask=torch.ones(6, 4),
        sample_mask=torch.ones(6),
        pair_index=torch.arange(3).repeat_interleave(2),
        is_chosen=torch.arange(6) % 2 == 0,
    )
    cfg = MasterConfig.model_construct(
        policy={"train_micro_batch_size": 1, "train_global_batch_size": 8}
    )
    padded = next(
        add_ref_logprobs_to_data([batch], MockPolicy(torch.zeros(16, 4)), cfg)
    )
    loss_fn = DPOLossFn(DPOLossConfig(sft_loss_weight=sft_loss_weight))
    values = torch.linspace(-1.0, 0.5, 18).reshape(6, 3).requires_grad_()
    batch["reference_policy_logprobs"] = torch.zeros(6, 4)
    loss, metrics = loss_fn(
        values, batch, batch["sample_mask"].sum(), torch.tensor(18.0)
    )
    loss.backward()
    # Extreme filler scores expose accidental inclusion in either the loss or metrics.
    padded_values = torch.cat(
        [values.detach(), torch.full((10, 3), 100.0)]
    ).requires_grad_()
    padded_loss, padded_metrics = loss_fn(
        padded_values, padded, padded["sample_mask"].sum(), torch.tensor(18.0)
    )
    padded_loss.backward()
    torch.testing.assert_close(padded_loss, loss)
    assert padded_metrics == pytest.approx(metrics)
    torch.testing.assert_close(padded_values.grad[:6], values.grad)
    assert not padded_values.grad[6:].any()


def test_exit_on_max_steps(mock_dpo_components):
    """Test that training loop exits when max_num_steps is reached"""
    # Set max steps to 12, which is less than len(train_dataloader) * max_num_epochs
    mock_dpo_components["master_config"].dpo.max_num_steps = 12

    dpo_save_state = _initial_dpo_save_state()

    # Run training
    dpo_train(
        mock_dpo_components["policy"],
        mock_dpo_components["train_dataloader"],
        mock_dpo_components["val_dataloader"],
        mock_dpo_components["tokenizer"],
        mock_dpo_components["loss_fn"],
        mock_dpo_components["master_config"],
        mock_dpo_components["logger"],
        mock_dpo_components["checkpointer"],
        dpo_save_state,
    )

    # Verify we only trained for 12 steps.
    assert mock_dpo_components["policy"].train.call_count == 12
    final_timing_call = [
        call
        for call in mock_dpo_components["logger"].log_metrics.call_args_list
        if call.kwargs.get("prefix") == "timing/train"
    ][-1]
    assert final_timing_call.kwargs["step_finished"] is True


def test_configured_stop_forces_checkpoint_without_shortening_run(
    mock_dpo_components, capsys
):
    cfg = mock_dpo_components["master_config"]
    cfg.dpo = DPOConfig.model_validate(
        {
            **cfg.dpo.model_dump(),
            "max_num_steps": 100,
            "max_num_epochs": 10,
            "stop_after_step": 3,
        }
    )
    cfg.checkpointing.enabled = True
    cfg.checkpointing.save_period = 100
    cfg.checkpointing.metric_name = None

    checkpointer = mock_dpo_components["checkpointer"]
    checkpointer.init_tmp_checkpoint.return_value = "/tmp/segment_ckpt/tmp_step"
    dpo_save_state = _initial_dpo_save_state()

    with patch("nemo_rl.algorithms.dpo.torch.save"):
        dpo_train(
            mock_dpo_components["policy"],
            mock_dpo_components["train_dataloader"],
            mock_dpo_components["val_dataloader"],
            mock_dpo_components["tokenizer"],
            mock_dpo_components["loss_fn"],
            cfg,
            mock_dpo_components["logger"],
            checkpointer,
            dpo_save_state,
        )

    assert cfg.dpo.max_num_steps == 100
    assert mock_dpo_components["policy"].train.call_count == 3
    assert [c.args[0] for c in checkpointer.init_tmp_checkpoint.call_args_list] == [3]
    assert dpo_save_state.total_steps == 3
    assert "Configured stop-after step 3 has been reached" in capsys.readouterr().out


def test_configured_stop_already_reached_does_not_retrain(mock_dpo_components, capsys):
    cfg = mock_dpo_components["master_config"]
    cfg.dpo = DPOConfig.model_validate(
        {
            **cfg.dpo.model_dump(),
            "max_num_steps": 100,
            "max_num_epochs": 10,
            "stop_after_step": 3,
        }
    )
    dpo_save_state = _initial_dpo_save_state()
    dpo_save_state.total_steps = 3

    dpo_train(
        mock_dpo_components["policy"],
        mock_dpo_components["train_dataloader"],
        mock_dpo_components["val_dataloader"],
        mock_dpo_components["tokenizer"],
        mock_dpo_components["loss_fn"],
        cfg,
        mock_dpo_components["logger"],
        mock_dpo_components["checkpointer"],
        dpo_save_state,
    )

    mock_dpo_components["policy"].train.assert_not_called()
    mock_dpo_components["checkpointer"].init_tmp_checkpoint.assert_not_called()
    assert dpo_save_state.total_steps == 3
    assert "Configured stop-after step 3 was already reached" in capsys.readouterr().out


def test_exit_on_max_epochs(mock_dpo_components):
    """Test that training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    mock_dpo_components["master_config"].dpo.max_num_epochs = 2
    mock_dpo_components["master_config"].dpo.max_num_steps = 100

    dpo_save_state = _initial_dpo_save_state()

    # Run training
    dpo_train(
        mock_dpo_components["policy"],
        mock_dpo_components["train_dataloader"],
        mock_dpo_components["val_dataloader"],
        mock_dpo_components["tokenizer"],
        mock_dpo_components["loss_fn"],
        mock_dpo_components["master_config"],
        mock_dpo_components["logger"],
        mock_dpo_components["checkpointer"],
        dpo_save_state,
    )

    # Verify we trained for exactly two epochs (20 batches).
    assert mock_dpo_components["policy"].train.call_count == 20


def test_exit_on_timeout(mock_dpo_components, capsys, tmp_path):
    """Test that training loop exits when timeout is reached"""
    # Set max steps and epochs to large numbers
    mock_dpo_components["master_config"].dpo.max_num_steps = 100
    mock_dpo_components["master_config"].dpo.max_num_epochs = 10
    mock_dpo_components["master_config"].checkpointing.enabled = True
    mock_dpo_components["master_config"].checkpointing.metric_name = None
    mock_dpo_components["checkpointer"].init_tmp_checkpoint.return_value = str(
        tmp_path / "tmp_step"
    )

    dpo_save_state = _initial_dpo_save_state()

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with (
        patch("nemo_rl.algorithms.dpo.torch.save"),
        patch("nemo_rl.algorithms.dpo.TimeoutChecker") as mock_timeout_class,
    ):
        mock_timeout_instance = MagicMock()
        # Create a side_effect that returns False 7 times, then True
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

        # Run training
        dpo_train(
            mock_dpo_components["policy"],
            mock_dpo_components["train_dataloader"],
            mock_dpo_components["val_dataloader"],
            mock_dpo_components["tokenizer"],
            mock_dpo_components["loss_fn"],
            mock_dpo_components["master_config"],
            mock_dpo_components["logger"],
            mock_dpo_components["checkpointer"],
            dpo_save_state,
        )

        # Verify training stopped at 8 steps (when check_save returned True)
        assert mock_dpo_components["policy"].train.call_count == 8
        assert (
            mock_dpo_components["policy"].save_checkpoint.call_args.kwargs[
                "is_final_checkpoint"
            ]
            is False
        )

        # Verify the timeout message was printed and is near the end (not followed by more training)
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")

        # Find the timeout message
        timeout_line_idx = None
        for i, line in enumerate(output_lines):
            if "Timeout has been reached, stopping training early" in line:
                timeout_line_idx = i
                break

        assert timeout_line_idx is not None, "Timeout message not found in output"

        # Verify no new epoch started after timeout (which would indicate a bug where break was used instead of return)
        remaining_lines = output_lines[timeout_line_idx:]
        for line in remaining_lines:
            assert "Epoch" not in line or "Epoch 1/10" in line, (
                f"Training continued to next epoch after timeout: {line}"
            )


def test_ft_save_period_triggers_periodic_saves(mock_dpo_components):
    """ft_save_period triggers checkpoint saves independent of save_period."""
    cfg = mock_dpo_components["master_config"]
    cfg.dpo.val_period = 0
    cfg.dpo.max_num_steps = 5
    cfg.dpo.max_num_epochs = 1
    cfg.checkpointing.enabled = True
    cfg.checkpointing.save_period = 100  # only the final step would save
    cfg.checkpointing.ft_save_period = 2
    cfg.checkpointing.metric_name = None

    checkpointer = mock_dpo_components["checkpointer"]
    checkpointer.init_tmp_checkpoint.return_value = "/tmp/ft_ckpt_test/tmp_step"

    dpo_save_state = _initial_dpo_save_state()

    with patch("nemo_rl.algorithms.dpo.torch.save"):
        dpo_train(
            mock_dpo_components["policy"],
            mock_dpo_components["train_dataloader"],
            mock_dpo_components["val_dataloader"],
            mock_dpo_components["tokenizer"],
            mock_dpo_components["loss_fn"],
            cfg,
            mock_dpo_components["logger"],
            checkpointer,
            dpo_save_state,
        )

    # ft_save_period=2 -> steps 2, 4; save_period=100 contributes only the last step (5).
    saved_steps = [c.args[0] for c in checkpointer.init_tmp_checkpoint.call_args_list]
    assert saved_steps == [2, 4, 5]
