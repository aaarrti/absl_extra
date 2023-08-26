from __future__ import annotations

from typing import no_type_check

import clu.metrics
import clu.periodic_actions
import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, Int, jaxtyped


class UncheckedReportProgress(clu.periodic_actions.ReportProgress):
    def __call__(self, step: int, **kwargs) -> bool:
        return super().__call__(int(step))

    @no_type_check
    def _init_and_check(self, step: int, t: float):
        """Initializes and checks it was called at every step."""
        if self._previous_step is None:
            self._previous_step = step
            self._previous_time = t
            self._last_step = step
        else:
            self._last_step = step


class UncheckedPeriodicCallback(clu.periodic_actions.PeriodicCallback):
    def __call__(self, step: int, *args, **kwargs) -> bool:
        return super().__call__(int(step), *args, **kwargs)

    @no_type_check
    def _init_and_check(self, step: int, t: float):
        """Initializes and checks it was called at every step."""
        if self._previous_step is None:
            self._previous_step = step
            self._previous_time = t
            self._last_step = step
        else:
            self._last_step = step


@struct.dataclass
class NanSafeAverage(clu.metrics.Average):
    def compute(self) -> float:
        if self.count != 0:
            return super().compute()
        else:
            return 0


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
    ) -> F1Score:
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
            true_positive=0,
            false_positive=0,
            false_negative=0,
        )

    def compute(self) -> float:
        precision = nan_div(self.true_positive, self.true_positive + self.false_positive)
        recall = nan_div(self.true_positive, self.true_positive + self.false_negative)

        # Ensure we don't divide by zero if both precision and recall are zero
        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


@jaxtyped
@struct.dataclass
class BinaryAccuracy(NanSafeAverage):
    @classmethod
    def from_model_output(  # noqa
        cls,
        *,
        logits: Float[Array, "batch classes"],
        labels: Int[Array, "batch classes"],
        threshold: float = 0.5,
        **kwargs,
    ) -> BinaryAccuracy:
        predicted = jnp.asarray(logits >= threshold, logits.dtype)
        return super().from_model_output(values=jnp.asarray(predicted == labels, predicted.dtype))


def nan_div(a: float, b: float) -> float:
    if b == 0:
        return 0
    else:
        return a / b
