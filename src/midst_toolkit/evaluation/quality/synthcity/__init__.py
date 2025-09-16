"""
This Module reproduces the code necessary to run the Synthcity alpha precision evaluation computations. The files are
pulled and modified from this repository https://github.com/vanderschaarlab/synthcity.

There are two reasons for this port.

1: Synthcity, in its larger form, is incompatible with newer versions of PyTorch, beyond 2.4, which is limiting.

2: On Mac OS, every time its imported, some external python process is kicked off. This manifests as a launcher icon
appearing temporarily in the dock and then disappearing. If you're using VS code, it periodically re-imports the
library in the background (for whatever reason), resulting in the launcher icon appearing repeatedly and endlessly.

We don't need much of the Synthcity library for our evals. So having the computations reproduced here locally
alleviates both issues. We can re-evaluate this port with newer versions of Synthcity, if they come.
"""
