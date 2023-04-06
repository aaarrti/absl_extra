### ABSL-Extra

A collection of utils I commonly use for running my experiments.

Minimal example
```python
from absl import flags
from absl_extra import App, SlackNotifier
from ml_collections import ConfigDict

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", default=32, help=None)

def main(cmd: str, config: ConfigDict):
    pass

if __name__ == "__main__":
    app = App(
        notifier=SlackNotifier(),
        env_file="../.env",
        config_file="config.py"
   )
    app.run(main)
```
