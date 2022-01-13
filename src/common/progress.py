import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from rich import filesize
from rich.console import RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID
from rich.progress_bar import ProgressBar
from rich.text import Text


class CustomBarColumn(BarColumn):
    """Overrides ``BarColumn`` to provide support generators."""

    def render(self, task: Task) -> ProgressBar:
        """Gets a progress bar widget for a task."""
        return ProgressBar(
            total=max(0, task.total),
            completed=max(0, task.completed),
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=not task.started or math.isinf(task.remaining),
            animation_time=task.get_time(),
        )


@dataclass
class CustomInfiniteTask(Task):
    """Overrides ``Task`` to define an infinite task."""

    @property
    def time_remaining(self) -> Optional[float]:
        """Returns None for time remaining, following what PyTorch Lightning did."""
        return  # type: ignore[return-value] # noqa: WPS324


class CustomProgress(Progress):
    """Overrides ``Progress`` to support adding tasks that have an infinite total size."""

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: float = 100.0,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> TaskID:
        """Create task, with support for possible infinite tasks."""
        if not math.isfinite(total):
            task = CustomInfiniteTask(
                self._task_index,
                description,
                total,
                completed,
                visible=visible,
                fields=fields,
                _get_time=self.get_time,
                _lock=self._lock,
            )
            return self.add_custom_task(task)
        return super().add_task(description, start, total, completed, visible, **fields)

    def add_custom_task(self, task: CustomInfiniteTask, start: bool = True) -> TaskID:
        """Create custom task."""
        with self._lock:
            self._tasks[self._task_index] = task
            if start:
                self.start_task(self._task_index)
            new_task_index = self._task_index
            self._task_index = TaskID(int(self._task_index) + 1)
        self.refresh()
        return new_task_index


class CustomTimeColumn(ProgressColumn):
    """Custom time column, as in Pytorch Lightning."""

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def render(self, task: Task) -> Text:
        """Render time."""
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining
        elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
        remaining_delta = (
            "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
        )
        return Text(f"{elapsed_delta} • {remaining_delta}", style="progress.remaining")


class BatchesProcessedColumn(ProgressColumn):
    """Counter for processed."""

    def render(self, task: Task) -> RenderableType:
        """Render number of processed instances."""
        total = "--" if task.total == float("inf") else task.total
        return Text(f"{int(task.completed)}/{total}", style="progress.elapsed")


class ProcessingSpeedColumn(ProgressColumn):
    """Column for processing speed."""

    def render(self, task: Task) -> RenderableType:
        """Render processing speed."""
        task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        return Text(f"{task_speed}it/s", style="progress.data.speed")


class CustomFileSizeColumn(ProgressColumn):
    """Column for current file size of file."""

    # Only refresh once every two seconds because it doesn't need to be faster.
    max_refresh = 2

    def render(self, task: Task) -> RenderableType:
        """Render file size if filename is given in fields."""
        filepath: Optional[Path] = task.fields.get("filepath", None)

        if filepath and filepath.exists():
            file_size = filesize.decimal(int(filepath.stat().st_size))
            return Text(f"Size {file_size}", style="progress.filesize")

        return Text("")


def get_progress() -> Progress:
    """Get custom progress with support for generators."""
    return CustomProgress(
        "[progress.description]{task.description}",
        CustomBarColumn(),
        BatchesProcessedColumn(),
        CustomTimeColumn(),
        ProcessingSpeedColumn(),
        CustomFileSizeColumn(),
    )
