### ABSL-Extra

A collection of utils I commonly use for running my experiments.
It will:
- Notify on execution start, finish or failed.
  - By default, Notifier will just log those out to `stdout`.
  - I prefer receiving those in Slack, though (see example below).
- Log parsed CLI flags from `absl.flags.FLAGS` and config values from `config_file:get_config()`
- Inject `ml_collections.ConfigDict` from `config_file`, if kwargs provided.
- Inject `pymongo.collection.Collection` if `mongo_config` kwargs provided.

Minimal example
```python
from os import environ
from pymongo.collection import Collection
from absl import flags
from ml_collections import ConfigDict
from absl_extra import App, SlackNotifier, MongoConfig


FLAGS = flags.FLAGS
flags.DEFINE_integer("some_flag", default=4, help=None)

def main(cmd: str, config: ConfigDict, db: Collection) -> None: ...

if __name__ == "__main__":
    app = App(
        notifier=SlackNotifier(slack_token=environ["SLACK_TOKEN"], channel_id=environ["CHANNEL_ID"]),
        config_file="config.py",
        mongo_config=MongoConfig(uri=environ["MONGO_URI"], db_name="my_project", collection="experiment_1"),
   )
    app.run(main)
```
