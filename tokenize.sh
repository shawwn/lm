set -ex
PYTHONPATH=src exec python3 -m lm.cli.main encode "$@"
