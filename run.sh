#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

MONOREPO_ROOT="$(git rev-parse --show-toplevel)"
RANDOM_DIR="$(mktemp -d)"
cd "${MONOREPO_ROOT}"


usage() {
  local program
  program=$(basename "$0")
  cat <<HELP
  Run data generation and training.

  Usage: ${program} <step> [--options]

  Step can be:
    {generate|train|all}

  Examples
    ${program} generate
    ${program} train
    ${program} all
HELP
}


train() {
  echo "Training..."
  echo

  python "${MONOREPO_ROOT}/main.py"

  echo
  echo "🎉 Success 🎉"
  deactivate
}


generate() {
  echo "Generating data..."
  echo

  python "${MONOREPO_ROOT}/data/generator.py"

  echo
  echo "🎉 Success 🎉"
  deactivate
}


all() {
  echo "Running all..."
  echo

  generate
  train
}

main() {
  local command_name="${1:-help}"
  shift

  if [[ ("$#" -ne 0 ) ]]; then
    usage
    exit 1
  fi

  case $command_name in
  help)
    usage
    ;;
  generate)
    generate
    ;;
  train)
    train
    ;;
  all)
    all
    ;;
  *)
    echo "Unknown command ${command_name}!"
    usage
    exit 1
    ;;
  esac
}

main "$@"