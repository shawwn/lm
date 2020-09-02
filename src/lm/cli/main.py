"""Console script for lm."""
import argparse
import sys
from absl import app

def main(_):
    """Console script for lm."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "lm.cli.main")
    return 0

def run():
    app.run(main)
