from typing import no_type_check

import clu.metrics
import clu.periodic_actions
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


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

    true_positive: np.float32
    false_positive: np.float32
    false_negative: np.float32

    @classmethod
    @no_type_check
    def from_model_output(
        cls,
        *,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        threshold: float = 0.5,
        **kwargs,
    ) -> "F1Score":
        """

        Parameters
        ----------
        logits:
            2D float
        labels:
            2D int
        threshold
        kwargs

        Returns
        -------

        """
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
    @no_type_check
    def empty(cls) -> "F1Score":
        return F1Score(
            true_positive=jnp.asarray(0),
            false_positive=jnp.asarray(0),
            false_negative=jnp.asarray(0),
        )

    @no_type_check
    def compute(self) -> np.float32:
        precision = self.true_positive / (self.true_positive + self.false_positive)
        recall = self.true_positive / (self.true_positive + self.false_negative)

        # Ensure we don't divide by zero if both precision and recall are zero
        if precision + recall == 0:
            return jnp.asarray(0.0, self.true_positive.dtype)

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


@struct.dataclass
class BinaryAccuracy(clu.metrics.Average):
    @classmethod
    def from_model_output(  # noqa
        cls,
        *,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        threshold: float = 0.5,
        **kwargs,
    ) -> "BinaryAccuracy":
        """

        Parameters
        ----------
        logits:
            2D floats
        labels:
            2d int
        threshold
        kwargs

        Returns
        -------

        """
        predicted = jnp.asarray(logits >= threshold, logits.dtype)
        return super().from_model_output(values=jnp.asarray(predicted == labels, predicted.dtype))
