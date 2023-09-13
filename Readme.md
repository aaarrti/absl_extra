### ABSL-Extra

A collection of utils I commonly use for running my experiments.
It will:
- Notify on execution start, finish or failed.
  - By default, Notifier will just log those out to `stdout`.
  - I prefer receiving those in Slack, though (see example below).
- Log parsed CLI flags from `absl.flags.FLAGS` and config values from `config_file:get_config()`
- Select registered task to run based on --task= CLI argument.

Minimal example

```python
import os
from absl import logging
import tensorflow as tf

from absl_extra import tf_utils, tasks, notifier


@tasks.register_task(
    notifier=notifier.SlackNotifier(slack_token=os.environ["SLACK_BOT_TOKEN"], channel_id=os.environ["CHANNEL_ID"])
)
@tf_utils.requires_gpu
def main() -> None:
    if tf_utils.supports_mixed_precision():
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    
    with tf_utils.make_gpu_strategy().scope():
        logging.info("Doing some heavy lifting...")


if __name__ == "__main__":
    tasks.run()
```

#### `flax_utils.py`
- Common utilities used for training flax models, which I got tired of copy-pasting in every project.