'''
module with custom progress bar
'''

from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           ProgressColumn, SpinnerColumn, Text, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)


class SpeedColumn(ProgressColumn):
    '''speed column for progress bar'''

    def render(self, task: 'Task') -> Text:
        if task.speed is None:
            return Text('- it/s', style='red')
        return Text(f'{task.speed:.2f} it/s', style='red')


progress_bar = Progress(
    TextColumn('[bold]{task.description}'),
    SpinnerColumn('simpleDots'),
    TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn('|'),
    SpeedColumn(),
    TextColumn('|'),
    TimeElapsedColumn(),
    TextColumn('|'),
    TimeRemainingColumn(),
)
