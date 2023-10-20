import chex
import jax
import jax.numpy as jnp
import pytest

from absl_extra.clu_utils import BinaryAccuracy

PRNG_SEED = 69
BATCH_SIZE = 8
NUM_CLASSES = 5


def_y_true = jnp.ones([BATCH_SIZE, NUM_CLASSES], jnp.int32)
def_y_pred = jax.random.uniform(jax.random.PRNGKey(PRNG_SEED), [BATCH_SIZE, NUM_CLASSES], jnp.float32)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (jnp.ones_like(def_y_true), jnp.zeros_like(def_y_true, jnp.float32), 0.0),
        (jnp.ones_like(def_y_true), jnp.ones_like(def_y_true, jnp.float32), 1.0),
        (jnp.zeros_like(def_y_true), jnp.zeros_like(def_y_true, jnp.float32), 1.0),
    ],
    ids=["0%", "all 1s", "all 0s"],
)
def test_binary_accuracy(y_true, y_pred, expected):
    expected = jnp.asarray(expected, jnp.float32)
    acc = BinaryAccuracy.from_model_output(
        logits=y_pred,
        labels=y_true,
    ).compute()

    chex.assert_rank(acc, 0)
    chex.assert_trees_all_close(acc, expected, atol=0.01)
