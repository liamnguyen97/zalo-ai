import subprocess
import sys
import textwrap

import pytest

from datamate.sandbox.memory import OOM_HINT, is_oom


def test_is_oom():
    assert is_oom(MemoryError()) and is_oom(RuntimeError("Unable to allocate 40 GiB for an array"))
    assert not is_oom(KeyError("x"))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is enforced on Linux only")
def test_runaway_block_is_reported_not_fatal():
    """A block that allocates far beyond the cap comes back as an error for the Coder; the process lives on."""
    code = textwrap.dedent('''
        from datamate.sandbox.memory import apply_limit, OOM_HINT
        from datamate.sandbox.runner import Runner
        assert apply_limit(2) == 2
        r = Runner({"tables": {}})
        bad = r.run("import numpy as np\\nx = np.ones(10**10)\\nanswer = int(x.sum())")
        assert bad.error == OOM_HINT, bad.error
        ok = r.run("print(sum(range(5)))")
        assert ok.error is None and ok.output.strip() == "10", (ok.error, ok.output)
        print("ok")
    ''')
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr[-800:]
