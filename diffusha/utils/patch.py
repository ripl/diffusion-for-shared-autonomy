#!/usr/bin/env python3

"""Patch multiprocessing.arbitrary_address in python 3.9 that is known to cause Address already in use error.

https://github.com/SeldonIO/seldon-core/issues/3720#issuecomment-1313212063
"""

import tempfile
from multiprocessing import connection, util
from multiprocessing.connection import _mmap_counter


def arbitrary_address(family):
    """
    Return an arbitrary free address for the given family
    """
    if family == "AF_INET":
        return ("localhost", 0)
    elif family == "AF_UNIX":
        return tempfile.mktemp(prefix="listener-", dir=util.get_temp_dir())
    elif family == "AF_PIPE":
        return tempfile.mktemp(
            prefix=r"\\.\pipe\pyc-%d-%d-" % (os.getpid(), next(_mmap_counter)), dir=""
        )
    else:
        raise ValueError("unrecognized family")


connection.arbitrary_address = arbitrary_address
