# Contributing to Modalities
## Table of Contents

- [Getting Started](#getting-started)
- [Contribution Guidelines](#contribution-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Submitting a Merge Request](#submitting-a-merge-request)

## Getting started

Install the project via
```shell
# ensure that you already created and activated a virtual environment before
cd modalities && pip install .
```

For developers, use
```shell
# ensure that you already created and activated a virtual environment before
cd modalities && pip install -e .[tests,linting]
pre-commit install --install-hooks
```

## Contribution Guidelines

- Make sure to create a new branch for your changes. Branch names should be aligned with the conventional commit syntax, such as "feat/my-new-feature" or "refactor/component-x."

- Write clear and concise commit messages that adhere to conventional commit guidelines (see [Commit Guidelines](#commit-guidelines)).

- Provide unittests for your added functionality. In case you are doing something, which is not obvious, add comments/docs or cleaner code for transparency. If your changes are highly complex, seek the conversation with the team to explain it. E.g. using a Merge Request thread.

- Make sure your code passes all the tests and pre-commit hooks. Use `pytest` from within the root of your local repository.

- For vscode users, disable pytest coverage in `settings.json` to enable pytest debugging: `"python.testing.pytestArgs": ["--no-cov"]`

## Commit Guidelines

We follow the [conventional commit style](https://www.conventionalcommits.org/en/v1.0.0/) for our commit messages.  
The available commit types vary slightly from the official ones:
- `feat`: [see here](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)
- `fix`: [see here](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)
- `ci`: [see here](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)
- `docs`: [see here](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)
- `perf`: [see here](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)
- `test`: [see here](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)
- `refactor`: [see here](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)
- `debug`: temporary changes, which should in the best case stay reverted. Can also get used to trigger the CI for debugging. 
- `revert`: technical type, which is rather used when applying `git revert`. Can ofc. also get used manually when "reverting" a previous, typed change. Format should be then `revert: feat: my unstable feature`
- `chore`: rather flexible type. Some definitions are e.g. "Other changes that don't modify src or test files". People might tend to use this as leftover option for unclassifiable commits. This is not the intention, but instead always trying to categorize the own commits as good as possible into the other types.


## Submitting a Merge Request

1. Create a merge request onto the `main` branch.

2. Assign two reviewers, one with the regular reviewer functionality and one by using a comment e.g. `/cc @second-reviewer - reviewer`. If wished assign more people like this, but consider also marking "optional" reviewers to not block your merge request for too long.

3. If your Merge Request stays open and the target branch (e.g. `main`) receives updates in the meantime, which lead to conflicts with your changes, resolve those and request another review from the team members of your choice. Do this explicitly by re-assigning them according to `2.`
