from datetime import timedelta

from rich.console import RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.text import Text


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
        return Text(f"{int(task.completed)}/{total}", style="progress.download")


class ProcessingSpeedColumn(ProgressColumn):
    """Column for processing speed."""

    def render(self, task: Task) -> RenderableType:
        """Render processing speed."""
        task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        return Text(f"{task_speed}it/s", style="progress.data.speed")


def get_progress() -> Progress:
    """Get custom progress with support for generators."""
    return Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        BatchesProcessedColumn(),
        CustomTimeColumn(),
        ProcessingSpeedColumn(),
        TextColumn("[purple]{task.fields[comment]}[/]"),
    )
