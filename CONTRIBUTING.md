# Contributing to MIDST Toolkit

Thanks for your interest in contributing to the MIDST Toolkit!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Pre-commit hooks

```bash
pre-commit install
```

To run the checks, some of which will automatically re-format your code to fit the standards, you can run
```bash
pre-commit run --all-files
```
It can also be run on a subset of files by omitting the `--all-files` option and pointing to specific files or folders.

If you're using VS Code for development, pre-commit should setup git hooks that execute the pre-commit checks each
time you check code into your branch through the integrated source-control as well. This will ensure that each of your
commits conform to the desired format before they are run remotely and without needing to remember to run the checks
before pushing to a remote. If this isn't done automatically, you can find instructions for setting up these hooks
manually online.

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For code documentation, we try to adhere to the Google docstring style
(See [here](https://google.github.io/styleguide/pyguide.html), Section: Comments and Doc-strings). The implementation
of an extensive set of comments for the code in this repository is a work-in-progress. However, we are continuing to
work towards a better commented state for the code. For development, as stated in the style guide,
__any non-trivial or non-obvious methods added to the library should have a doc string__. For our library this
applies only to code added to the main library in `midst_toolkit`. Examples, research code, and tests need not
incorporate  the strict rules of documentation, though clarifying and helpful comments in that code is also
__strongly encouraged__.

> [!NOTE]
> As a matter of convention choice, classes are documented through their `__init__` functions rather than at the
> "class" level.

If you are using VS Code a very helpful integration is available to facilitate the creation of properly formatted
doc-strings called autoDocstring [VS Code Page](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
and [Documentation](https://github.com/NilsJPWerner/autoDocstring). This tool will automatically generate a docstring
template when starting a docstring with triple quotation marks (`"""`). To get the correct format, the following
settings should be prescribed in your VS Code settings JSON:

```json
{
    "autoDocstring.customTemplatePath": "",
    "autoDocstring.docstringFormat": "google",
    "autoDocstring.generateDocstringOnEnter": true,
    "autoDocstring.guessTypes": true,
    "autoDocstring.includeExtendedSummary": false,
    "autoDocstring.includeName": false,
    "autoDocstring.logLevel": "Info",
    "autoDocstring.quoteStyle": "\"\"\"",
    "autoDocstring.startOnNewLine": true
}
```

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including
[flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which
you need to fix before submitting a PR.

Last but not the least, we use type hints in our code which is then checked using
[mypy](https://mypy.readthedocs.io/en/stable/).

**Note**: We use the modern mypy types introduced in Python 3.10 and above. See some of the
[documentation here](https://mypy.readthedocs.io/en/stable/builtin_types.html)

For example, this means that we're using `list[str], tuple[int, int], tuple[int, ...], dict[str, int], type[C]` as
built-in types and `Iterable[int], Sequence[bool], Mapping[str, int], Callable[[...], ...]` from collections.abc
(as now recommended by mypy).

We also use the new Optional and Union specification style:
```python
Optional[typing_stuff] -> typing_stuff | None
Union[typing1, typing2] -> typing1 | typing2
Optional[Union[typing1, typing2]] -> typing1 | typing2 | None
```

There is a custom script that enforces this style. It is not infallible. So if there is an issue with it please fix or
report it to us.

## Tests

All tests for the library are housed in the tests folder. The unit and integration tests are run using `pytest`. These
tests are automatically run through GitHub integrations on PRs to the main branch of this repository. PRs that fail
any of the tests will not be eligible to be merged until they are fixed.

To run all tests in the tests folder one only needs to run (with the venv active)
```bash
pytest .
```
To run a specific test with pytest, one runs
```bash
pytest tests/checkpointing/test_best_checkpointer.py
```

If you use VS Code for development, you can setup the tests with the testing integration so that you can run
debugging and other IDE features. Setup will vary depending on your VS Code environment, but in your .vscode
folder your `settings.json` might look something like

``` JSON
{
    "python.testing.unittestArgs": [
        "-v",
        "-s",
        ".",
        "-p",
        "test_*.py"
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "."
    ]
}
```
