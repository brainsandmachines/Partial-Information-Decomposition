"""Small output/path helpers shared by PID plotting and simulation code."""


def safe_filename(name):
    """
    Return the filename exactly as given.
    Nothing is deleted or replaced.
    """

    return str(name)
