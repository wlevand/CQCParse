import sys
from typing import TextIO, Optional, Any
from contextlib import contextmanager
from rich import print as rich_print

level = 0

def _styled_print(label: str, color: str, *msgs: Any, file: Optional[TextIO] = None) -> None:
    """
    Generalized styled print for internal use. Used in other print functions
    Is a private function
    """
    target = file or PRINT_TARGET
    is_terminal = hasattr(target, 'isatty') and target.isatty()

    text_parts = [m if isinstance(m, str) else repr(m) for m in msgs]
    text = ' '.join(text_parts)

    if is_terminal:
        print(f"\033[{color}m[{label}] {text} \033[0m", file=target)
    else:
        print(f"[{label}] {text}", file=target)


def infoprint(*msgs: Any, file: Optional[TextIO] = None):
    _styled_print("INFO", "93", *msgs, file=file)

def debugprint(*msgs: Any, file: Optional[TextIO] = None):
    _styled_print("DEBUG", "92", *msgs, file=file)

def printtest(*msgs: Any, file: Optional[TextIO] = None):
    _styled_print("TESTPRINT", "92", *msgs, file=file)

def debugfunc(msgs: str, tag: str = "") -> None:
    if level >= 1:
        _styled_print(f"DEBUG][{tag}", "95", msgs)

def debug_deep(msgs: str, tag: str = "") -> None:
    if level >= 2:
        _styled_print(f"DEBUG DEEP][{tag}", "95", msgs)


PRINT_TARGET: TextIO = sys.stdout

@contextmanager
def use_print_target(temp_target: TextIO):
    """
    Temporarily set PRINT_TARGET inside a context.
    Instead of simply setting smth like open file 
    (it would need to be closed too)
    """
    global PRINT_TARGET
    prev_target = PRINT_TARGET
    try:
        PRINT_TARGET = temp_target
        yield
    finally:
        PRINT_TARGET = prev_target
        temp_target.close()


def separatorprint(*title: Any, file: Optional[TextIO] = None):
    """
    Marker of section, function...
    
    Print to specified file or global target; if file is None, then PRINT_TARGET.
    Won't print color modifiers if file is not None
    """
    target = file or PRINT_TARGET
    is_terminal = hasattr(target, 'isatty') and target.isatty()

    if not title:
        text = ''
    else:
        text_parts = []
        for t in title:
            if not isinstance(t, str):
                t = repr(t)
            text_parts.append(t)
        text = ' '.join(text_parts)
    if is_terminal:
        print('\n\033[95m'+'-'*15+f' {text} '+'-'*15+'\033[0m', file=target)
    else:
        print('\n'+'-'*15+f' {text} '+'-'*15, file=target)


def coolprint(text: str, file: Optional[TextIO] = None, colorinfo = 'italic yellow2'):
    """
    Print styled yellow italic text using rich to the given output stream.
    
    e.g.: colorinfo = 'italic yellow2' - rich library color specification
    
    https://rich.readthedocs.io/en/latest/introduction.html#quick-start
    
    Some options for text: bold italic yellow on red blink

    Also emoji:
    >>> from rich import print
    >>> print(":warning:")
    ⚠️

    """
    rich_print(f"[{colorinfo}]{text}[/{colorinfo}]", file=file or sys.stdout)
