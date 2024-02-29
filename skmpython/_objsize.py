from sys import getsizeof
from gc import get_referents
from types import ModuleType, FunctionType
# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj)->int:
    """Get the size of an object (including all sub-objects) in bytes.

    Args:
        - `obj (Any)`: Any object that is not `type`, `ModuleType`, `FunctionType`.

    Raises:
        - `TypeError`: getsize() does not take argument of type `type`, `ModuleType`, `FunctionType`.

    Returns:
        - `int`: Total size of the object.
    """
    if isinstance(obj, BLACKLIST):
        raise TypeError(
            'getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size