import jax.numpy as jnp
import jax
import chex

from absl_extra.jax_utils import binary_focal_crossentropy

PRNG_SEED = 69
BATCH_SIZE = 8
NUM_CLASSES = 5


def_y_true = jnp.ones([BATCH_SIZE, NUM_CLASSES], jnp.int32)
def_y_pred = jax.random.uniform(
    jax.random.PRNGKey(PRNG_SEED), [BATCH_SIZE, NUM_CLASSES], jnp.float32
)


def test_focal_loss():
    loss = binary_focal_crossentropy(def_y_pred, def_y_true)

    chex.assert_rank(loss, 1)
    chex.assert_shape(loss, (8,))
    chex.assert_type(loss, jnp.float32)
    chex.assert_tree_all_finite(loss)


def test_loss_converges_to_0():
    y_pred = jnp.asarray(def_y_true.astype(jnp.float32), jnp.float32)
    loss = binary_focal_crossentropy(
        y_pred,
        def_y_true,
    )
    chex.assert_trees_all_close(loss, 0, atol=0.01)
