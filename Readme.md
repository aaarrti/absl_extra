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
from pymongo.collection import Collection
from ml_collections import ConfigDict
from absl import logging
from absl_extra import register_task, SlackBaseNotifier, MongoConfig, run


@register_task
def main(config: ConfigDict, db: Collection) -> None:
    logging.info("Doing some heavy lifting...")


if __name__ == "__main__":
    run(
        config_file="config.py",
        mongo_config=MongoConfig(
            uri=os.environ["MONGO_URI"], db_name="my_project", collection="experiment_1"
        ),
        notifier=SlackBaseNotifier(
            slack_token=os.environ["SLACK_BOT_TOKEN"], channel_id=os.environ["CHANNEL_ID"]
        ),
    )
```
