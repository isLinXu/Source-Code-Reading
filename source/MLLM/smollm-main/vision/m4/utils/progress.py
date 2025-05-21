import os
import time


M4_DISABLE_RICH = False
if os.environ.get("M4_DISABLE_RICH", "") == "1":
    M4_DISABLE_RICH = True
else:
    try:
        import rich  # noqa
    except ModuleNotFoundError:
        M4_DISABLE_RICH = True

if not M4_DISABLE_RICH:
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TimeElapsedColumn

else:
    # This is a simple equivalent of some of the `rich`'s classes we use but which doesn't use
    # `rich` or any formatting. We use it if there is no `rich` installed or during HPC training
    # where we don't use a live console and log to a file instead - so we want easy to read logs and
    # `rich`'s output mangling causes more trouble than it helps.

    class BarColumn:
        def render(self, task):
            return ""

    class MofNCompleteColumn:
        def render(self, task):
            if task.total_steps is not None:
                total_steps = task.total_steps
            else:
                total_steps = "UNK"
            return f"{task.completed}/{total_steps}"

    class TaskProgressColumn:
        def render(self, task):
            if task.total_steps is not None:
                percent = int(task.completed / task.total_steps * 100)
                return f"{percent:>3}%"
            else:
                return "UNK%"

    class TimeElapsedColumn:
        def render(self, task):
            time_diff = time.gmtime(time.time() - task.start_time)
            days = int(time.strftime("%j", time_diff)) - 1
            time_str = time.strftime("%H:%M:%S", time_diff)
            return f"{days}:{time_str}"

    class Task:
        def __init__(self, description, total_steps, *args, **kwargs):
            self.description = description
            self.total_steps = total_steps

            self.completed = 0
            self.start_time = time.time()

        def step(self, advance_steps):
            self.completed += advance_steps

    class Progress:
        def __init__(self, *args, **kwargs):
            self.tasks = []
            self.description = "Progress"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, task, advance):
            task.step(advance)
            return self

        def add_task(self, description, total, *args, **kwargs):
            task = Task(description, total)
            self.tasks.append(task)
            return task
