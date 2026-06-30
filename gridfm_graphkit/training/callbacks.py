from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import os
import shutil
import tempfile
import torch


class SaveBestModelStateDict(Callback):
    def __init__(
        self,
        monitor: str,
        mode: str = "min",
        filename: str = "best_model_state_dict.pt",
    ):
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_score = float("inf") if mode == "min" else -float("inf")

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return  # Metric not available yet

        # Check if this is the best score so far
        if (self.mode == "min" and current < self.best_score) or (
            self.mode == "max" and current > self.best_score
        ):
            self.best_score = current

            logger = trainer.logger
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = os.path.join(tmp_dir, self.filename)
                torch.save(pl_module.state_dict(), model_path)

                # Log via MLflow's client API so it works with any artifact
                # store; fall back to copying into save_dir otherwise.
                if isinstance(logger, MLFlowLogger):
                    logger.experiment.log_artifact(
                        logger.run_id,
                        model_path,
                        artifact_path="model",
                    )
                else:
                    dest = os.path.join(logger.save_dir, "model")
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(model_path, dest)
