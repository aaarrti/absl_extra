#!/usr/bin/env zsh

set -ex

# shellcheck disable=SC2046
isort $(find absl_extra -iregex '.*\(py\)')
# shellcheck disable=SC2046
isort $(find tests -iregex '.*\(py\)')