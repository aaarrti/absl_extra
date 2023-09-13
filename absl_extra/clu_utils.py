import clu.metrics
import clu.periodic_actions
import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, Int32, jaxtyped


@jaxtyped
@struct.dataclass
class F1Score(clu.metrics.Metric):
    """
    Class F1Score
    This class represents the F1 Score metric for evaluating classification models.

    - A model will obtain a high F1 score if both Precision and Recall are high.
    - A model will obtain a low F1 score if both Precision and Recall are low.
    - A model will obtain a medium F1 score if one of Precision and Recall is low and the other is high.
    - Precision: Precision is a measure of how many of the positively classified examples were actually positive.
    - Recall (also called Sensitivity or True Positive Rate): Recall is a measure of how many of the actual positive
    examples were correctly labeled by the classifier.

    """

    true_positive: Float[Array, "1"]
    false_positive: Float[Array, "1"]
    false_negative: Float[Array, "1"]

    @classmethod
    def from_model_output(
        cls,
        *,
        logits: Float[Array, "batch classes"],  # noqa
        labels: Int32[Array, "batch classes"],  # noqa
        threshold: float = 0.5,
        **kwargs,
    ) -> "F1Score":
        probs = jax.nn.sigmoid(logits)
        predicted = jnp.asarray(probs >= threshold, labels.dtype)
        true_positive = jnp.sum((predicted == 1) & (labels == 1))
        false_positive = jnp.sum((predicted == 1) & (labels == 0))
        false_negative = jnp.sum((predicted == 0) & (labels == 1))

        return F1Score(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )

    def merge(self, other: "F1Score") -> "F1Score":
        return F1Score(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
        )

    @classmethod
    def empty(cls) -> "F1Score":
        return F1Score(
            true_positive=jnp.asarray(0),
            false_positive=jnp.asarray(0),
            false_negative=jnp.asarray(0),
        )

    def compute(self) -> Float[Array, "1"]:
        precision = self.true_positive / (self.true_positive + self.false_positive)
        recall = self.true_positive / (self.true_positive + self.false_negative)

        # Ensure we don't divide by zero if both precision and recall are zero
        if precision + recall == 0:
            return jnp.asarray(0.0, self.true_positive.dtype)

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


@jaxtyped
@struct.dataclass
class BinaryAccuracy(clu.metrics.Average):
    @classmethod
    def from_model_output(  # noqa
        cls,
        *,
        logits: Float[Array, "batch classes"],  # noqa
        labels: Int32[Array, "batch classes"],  # noqa
        threshold: float = 0.5,
        **kwargs,
    ) -> "BinaryAccuracy":
        predicted = jnp.asarray(logits >= threshold, logits.dtype)
        return super().from_model_output(values=jnp.asarray(predicted == labels, predicted.dtype))
