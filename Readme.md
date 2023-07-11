### ABSL-Extra

A collection of utils I commonly use for running my experiments.
It will:
- Notify on execution start, finish or failed.
  - By default, Notifier will just log those out to `stdout`.
  - I prefer receiving those in Slack, though (see example below).
- Log parsed CLI flags from `absl.flags.FLAGS` and config values from `config_file:get_config()`
- Inject `ml_collections.ConfigDict` from `config_file`, if kwarg provided.
- Inject `pymongo.collection.Collection` if `mongo_config` kwarg provided.

Minimal example

```python
import os
from pymongo.collection import Collection
from ml_collections import ConfigDict
from absl import logging
import tensorflow as tf

from absl_extra import tf_utils, tasks, notifier


@tasks.register_task(
    config_file="config.py",
    mongo_config=dict(uri=os.environ["MONGO_URI"], db_name="my_project", collection="experiment_1"),
    notifier=notifier.SlackNotifier(slack_token=os.environ["SLACK_BOT_TOKEN"], channel_id=os.environ["CHANNEL_ID"])
)
@tf_utils.requires_gpu
def main(config: ConfigDict, db: Collection) -> None:
    if tf_utils.supports_mixed_precision():
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    
    with tf_utils.make_gpu_strategy().scope():
        logging.info("Doing some heavy lifting...")


if __name__ == "__main__":
    tasks.run()
```


# Planned for:
- global app state for different tasks
- list of pre/post hooks 
- keras callback with notifier
