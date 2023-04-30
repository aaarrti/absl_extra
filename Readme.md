### ABSL-Extra

A collection of utils I commonly use for running my experiments.
It will:
- Notify on execution start, finish or failed.
  - By default, Notifier will just log those out to `stdout`.
  - I prefer receiving those in Slack, though (see example below).
- Log parsed CLI flags from `absl.flags.FLAGS` and config values from `config_file:get_config()`
- Inject `ml_collections.ConfigDict` from `config_file`, if kwarg provided.
- Inject `pymongo.collection.Collection` if `mongo_config` kwarg provided.
- `log_after` and `log_before` decorators, which proved extremely usefully for print-debugging.

Minimal example
```python
import os
import functools
from pymongo.collection import Collection
from absl import flags, app
from ml_collections import ConfigDict
from absl_extra import hook_main, SlackNotifier, MongoConfig


FLAGS = flags.FLAGS
flags.DEFINE_integer("some_flag", default=4, help=None)

@functools.partial(
    hook_main,
    app_name="some_name",
    config_file="config.py",
    mongo_config=MongoConfig(
        uri=os.environ["MONGO_URI"], db_name="my_project", collection="experiment_1"
    ),
    notifier=SlackNotifier(
        slack_token=os.environ["SLACK_BOT_TOKEN"], channel_id=os.environ["CHANNEL_ID"]
    ),
)
def main(cmd: str, config: ConfigDict, db: Collection) -> None:
    # Do all the heavy lifting. 
    pass

if __name__ == "__main__":
    app.run(main)
```
