# Releasing in Modalitites
This tutorial describes the procedure to release a new version of the Modalities package.

## Release Types
We follow the release types as defined by [Semantic Versioning](https://semver.org/). The version number is defined as `MAJOR.MINOR.PATCH` where:
- `MAJOR` is incremented when you make incompatible API changes,
- `MINOR` is incremented when you add functionality in a backwards-compatible manner, and
- `PATCH` is incremented when you make backwards-compatible bug fixes.


## Releasing a new Modalitites version
1. Update the version number in the pyproject.toml file.
2. Commit the version bump following the convention `git commit --no-verify -m "<version number>"`.
   The `--no-verify` flag is used to skip the pre-commit hooks.
3. Run `git push` to push the changes to the remote repository.
4. Make sure that all the tests pass in the main branch.
5. Tag the commit with the version number following the convention `git tag <version number>`.
6. Push the tag to the remote repository using `git push --tags`. Note, this command will push all the tags to the remote repository.
   This command triggers the [CI/CD pipeline](../../.github/workflows/release_automation.yml) to build and deploy the package to the PyPI repository.

