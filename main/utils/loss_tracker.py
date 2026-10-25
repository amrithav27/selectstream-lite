from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np

from main.utils.logging import get_logger

logger = get_logger("main.loss_tracker")


class LossTracker:
    """
    Tracks, records, and plots multi-task losses per video training step.
    Saves JSON/CSV logs and generates visualization charts after each epoch.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.step_counter = 0

    def log_video_loss(
        self,
        epoch: int,
        video_idx: int,
        video_id: str,
        stats: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Record the losses from a single video training iteration.
        """
        self.step_counter += 1

        def _to_float(val: Any, default: float = 0.0) -> float:
            if val is None:
                return default
            if hasattr(val, "item"):
                try:
                    return float(val.item())
                except Exception:
                    pass
            try:
                return float(val)
            except Exception:
                return default

        entry: Dict[str, Any] = {
            "step": self.step_counter,
            "epoch": epoch,
            "video_idx": video_idx,
            "video_id": str(video_id),
            "loss_total": _to_float(stats.get("loss_total")),
            "loss_ans": _to_float(stats.get("loss_ans")),
            "loss_ret": _to_float(stats.get("loss_ret")),
            "loss_ret_id": _to_float(stats.get("loss_ret_id")),
            "loss_ret_time": _to_float(stats.get("loss_ret_time")),
            "loss_spar": _to_float(stats.get("loss_spar")),
            "loss_importance": _to_float(stats.get("loss_importance")),
            "hit_id_at_k": _to_float(stats.get("hit_id_at_k")),
            "hit_time_at_k": _to_float(stats.get("hit_time_at_k")),
        }

        # Optional extras if present
        if "loss_spar_subgraph" in stats:
            entry["loss_spar_subgraph"] = _to_float(stats.get("loss_spar_subgraph"))
        if "loss_spar_redundancy" in stats:
            entry["loss_spar_redundancy"] = _to_float(stats.get("loss_spar_redundancy"))
        if "subgraph_size" in stats:
            entry["subgraph_size"] = _to_float(stats.get("subgraph_size"))

        self.history.append(entry)
        return entry

    def save_logs(self, epoch: Optional[int] = None) -> None:
        """
        Save the loss history to both JSON and CSV files.
        """
        # Save complete JSON
        json_path = os.path.join(self.output_dir, "loss_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if epoch is not None:
            epoch_json = os.path.join(self.output_dir, f"loss_history_epoch_{epoch}.json")
            epoch_entries = [e for e in self.history if e["epoch"] == epoch]
            with open(epoch_json, "w", encoding="utf-8") as f:
                json.dump(epoch_entries, f, indent=2)

        # Save CSV
        if self.history:
            csv_path = os.path.join(self.output_dir, "loss_history.csv")
            fieldnames = list(self.history[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.history)

    def plot_epoch_losses(
        self,
        epoch: int,
        smoothing_window: int = 5,
    ) -> Optional[str]:
        """
        Plot training losses across videos for the given epoch and save the figure.
        """
        if not self.history:
            logger.warning("No loss records found to plot.")
            return None

        # Filter entries for the current epoch (or all entries up to this epoch)
        epoch_entries = [e for e in self.history if e["epoch"] == epoch]
        if not epoch_entries:
            epoch_entries = self.history

        try:
            import matplotlib
            matplotlib.use("Agg")  # Headless backend (safe for servers/terminals)
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed. Skipping loss plot generation.")
            return None

        # Save logs first
        self.save_logs(epoch=epoch)

        videos = np.arange(1, len(epoch_entries) + 1)

        def _smooth(values: np.ndarray, window: int) -> np.ndarray:
            if len(values) < window or window <= 1:
                return values
            box = np.ones(window) / float(window)
            return np.convolve(values, box, mode="same")

        def _extract(key: str) -> np.ndarray:
            return np.array([e.get(key, 0.0) for e in epoch_entries], dtype=np.float32)

        loss_total = _extract("loss_total")
        loss_ans = _extract("loss_ans")
        loss_ret = _extract("loss_ret")
        loss_ret_id = _extract("loss_ret_id")
        loss_ret_time = _extract("loss_ret_time")
        loss_spar = _extract("loss_spar")
        loss_imp = _extract("loss_importance")
        hit_id = _extract("hit_id_at_k")
        hit_time = _extract("hit_time_at_k")

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"SelectStream Training Losses — Epoch {epoch} ({len(epoch_entries)} Videos)", fontsize=16, fontweight="bold")

        # 1. Total Loss
        ax = axs[0, 0]
        ax.plot(videos, loss_total, alpha=0.35, color="royalblue", label="Raw")
        ax.plot(videos, _smooth(loss_total, smoothing_window), color="royalblue", linewidth=2.0, label="Smoothed")
        ax.set_title("Total Loss (L_total)", fontweight="bold")
        ax.set_xlabel("Video Count")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # 2. Answer Generation Loss (Phrase CE)
        ax = axs[0, 1]
        ax.plot(videos, loss_ans, alpha=0.35, color="coral", label="Raw")
        ax.plot(videos, _smooth(loss_ans, smoothing_window), color="coral", linewidth=2.0, label="Smoothed")
        ax.set_title("Answer Generation Loss (L_ans: Cross-Entropy)", fontweight="bold")
        ax.set_xlabel("Video Count")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # 3. Information Score Loss (Importance BCE)
        ax = axs[0, 2]
        ax.plot(videos, loss_imp, alpha=0.35, color="mediumpurple", label="Raw")
        ax.plot(videos, _smooth(loss_imp, smoothing_window), color="mediumpurple", linewidth=2.0, label="Smoothed")
        ax.set_title("Information Score Loss (L_importance: Slot BCE)", fontweight="bold")
        ax.set_xlabel("Video Count")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # 4. Retrieval Loss Breakdown
        ax = axs[1, 0]
        ax.plot(videos, _smooth(loss_ret, smoothing_window), color="teal", linewidth=2.0, label="Ret Combined")
        ax.plot(videos, _smooth(loss_ret_id, smoothing_window), color="seagreen", linestyle="--", label="Ret ID")
        ax.plot(videos, _smooth(loss_ret_time, smoothing_window), color="darkorange", linestyle=":", label="Ret Time")
        ax.set_title("Retrieval Loss (L_ret)", fontweight="bold")
        ax.set_xlabel("Video Count")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # 5. Sparsity Loss
        ax = axs[1, 1]
        ax.plot(videos, loss_spar, alpha=0.35, color="crimson", label="Raw")
        ax.plot(videos, _smooth(loss_spar, smoothing_window), color="crimson", linewidth=2.0, label="Smoothed")
        ax.set_title("Sparsity Loss (L_spar)", fontweight="bold")
        ax.set_xlabel("Video Count")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # 6. Evidence Retrieval Hit@K
        ax = axs[1, 2]
        ax.plot(videos, _smooth(hit_id, smoothing_window), color="dodgerblue", linewidth=1.8, label="Hit ID @ K")
        ax.plot(videos, _smooth(hit_time, smoothing_window), color="forestgreen", linewidth=1.8, label="Hit Time @ K")
        ax.set_title("Evidence Retrieval Hit Rate @ K", fontweight="bold")
        ax.set_xlabel("Video Count")
        ax.set_ylabel("Hit Rate (0.0 - 1.0)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        epoch_plot_path = os.path.join(self.output_dir, f"loss_plots_epoch_{epoch}.png")
        latest_plot_path = os.path.join(self.output_dir, "loss_plots_latest.png")
        plt.savefig(epoch_plot_path, dpi=150)
        plt.savefig(latest_plot_path, dpi=150)
        plt.close(fig)

        logger.info(f"Loss plot saved to: {epoch_plot_path}")

        # Log summary statistics
        logger.info(
            f"[Epoch {epoch} Summary] "
            f"L_total: {loss_total.mean():.4f} | "
            f"L_ans: {loss_ans.mean():.4f} | "
            f"L_ret: {loss_ret.mean():.4f} | "
            f"L_imp: {loss_imp.mean():.4f} | "
            f"L_spar: {loss_spar.mean():.4f} | "
            f"Hit_time: {hit_time.mean():.3f}"
        )

        return epoch_plot_path
