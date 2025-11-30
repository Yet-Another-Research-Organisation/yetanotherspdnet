# Logging and formatting utilities
from typing import Callable, Iterable, Optional
from rich.progress import (
    SpinnerColumn,
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import logging
from logging import FileHandler
from rich.logging import RichHandler


from pathlib import Path


def setup_logging(
    level: str = "INFO",
    format: str = "%(message)s",
    markup: bool = True,
    file: Optional[str] = None,
) -> logging.Logger:
    """Setup logging to use rich handler
    Parameters
    ----------
    level : str, optional
        Logging level, by default "INFO"
    format : str, optional
        Format of the logging messages, by default "%(message)s"
    markup : bool, optional
        Whether to use rich markup, by default True
    file: str, optional
        File to additionally write the logs
    Returns
    -------
    logging.Logger
        Logger object
    """
    # Create formatter
    formatter = logging.Formatter(format, datefmt="[%X]")

    # Create RichHandler
    rich_handler = RichHandler(markup=markup, rich_tracebacks=True)
    rich_handler.setLevel(level)
    rich_handler.setFormatter(formatter)

    handlers = [rich_handler]

    if file is not None:
        Path(file).touch()
        # Create and configure FileHandler
        file_handler = FileHandler(file)
        file_handler.setLevel(level)
        # Use detailed format for file logging (time, level, message)
        file_format = "%(asctime)s - %(levelname)s - %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Get or create logger
    logger = logging.getLogger("rich")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def format_params(params: dict, markup: bool = True) -> str:
    """Format a dictionary of parameters for logging
    Parameters
    ----------
    params : dict
        Dictionary of parameters
    markup : bool, optional
        Whether to use rich markup, by default True
    Returns
    -------
    str
        Formatted string
    """
    string = ""
    for key, value in params.items():
        if markup:
            string += "    :black_circle_for_record: "
        else:
            string += "    - "
        string += f"{key}: {value}\n"
    return string


def setup_dataloading_iterator(
    iterations: Iterable,
    data_reading: Callable,
    verbose: Optional[int] = 0,
    task_description: Optional[str] = "",
) -> Iterable:
    """Setup a data loading iterator with optional rich progress bar visualization.

    Creates an iterator that applies a data reading function to each item in the
    input iterable. When verbose mode is enabled, displays a rich progress bar
    with detailed information including current item, completion percentage,
    elapsed time, and estimated remaining time.

    Parameters
    ----------
    iterations : Iterable
        An iterable containing items to be processed (e.g., file paths, indices)
    data_reading : Callable
        A function that takes a single item from iterations and returns processed data.
        Should accept one argument and return the loaded/processed data.
    verbose : int, optional
        Verbosity level controlling progress bar display, by default 0.
        If < 1, no progress bar is shown and a simple map is returned.
        If >= 1, displays a rich progress bar with detailed information.
    task_description : str, optional
        Description text to display in the progress bar, by default "".
        Only used when verbose >= 1.

    Returns
    -------
    Iterable
        An iterable that yields the results of applying data_reading to each
        item in iterations. If verbose < 1, returns a simple map object.
        If verbose >= 1, returns a generator with rich progress bar display.

    Notes
    -----
    The function converts the input iterations to a list to determine the total
    length for progress tracking, which may consume additional memory for large
    iterables.

    Examples
    --------
    >>> def load_file(path):
    ...     return np.load(path)
    >>> file_paths = ['file1.npy', 'file2.npy', 'file3.npy']
    >>>
    >>> # Simple iteration without progress bar
    >>> data_iter = setup_dataloading_iterator(file_paths, load_file, verbose=0)
    >>>
    >>> # With progress bar
    >>> data_iter = setup_dataloading_iterator(
    ...     file_paths, load_file, verbose=1, task_description="Loading data files"
    ... )
    >>> data_list = list(data_iter)
    """

    if verbose < 1:
        # No progress bar - simple map
        return map(data_reading, iterations)

    items_list = list(iterations)  # Convert to list to get length

    def progress_generator():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            TextColumn("Current: [bold blue]{task.fields[current_item]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                task_description, total=len(items_list), current_item="Starting..."
            )

            for item in items_list:
                result = data_reading(item)
                progress.update(task, advance=1, current_item=str(item))
                yield result

    return progress_generator()
