import copy
import yaml
import torch

from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.tasks.feature_reconstruction_task import FeatureReconstructionTask


# ponytail: complements test_full_pipeline::test_train (which only checks the
# train CLI doesn't crash) by asserting the model actually *learns* — gradients
# flow and the loss drops when we overfit a single batch. Runs on CPU because
# torch_geometric's scatter_reduce isn't implemented for MPS (see env-setup).
def test_training_overfits_single_batch():
    with open("tests/config/gridFMv0.1_dummy.yaml") as f:
        cfg = yaml.safe_load(f)
    # Bump LR so the decrease is unambiguous over a handful of steps.
    cfg["optimizer"]["learning_rate"] = 1e-3
    args = NestedNamespace(**cfg)

    torch.manual_seed(args.seed)

    dm = LitGridDataModule(args, data_dir="tests/data")

    class DummyTrainer:
        is_global_zero = True

    dm.trainer = DummyTrainer()
    dm.setup("fit")

    task = FeatureReconstructionTask(args, dm.node_normalizers, dm.edge_normalizers)
    task.configure_optimizers()  # sets task.optimizer
    task.train()

    pristine = next(iter(dm.train_dataloader()))

    def step():
        batch = copy.deepcopy(pristine)  # forward() masks x in place
        _, loss_dict = task.shared_step(batch)
        loss = loss_dict["loss"]
        task.optimizer.zero_grad()
        loss.backward()
        task.optimizer.step()
        return loss.item()

    first = step()
    for _ in range(20):
        last = step()

    assert torch.isfinite(torch.tensor(first))
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
