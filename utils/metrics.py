import numpy as np


def accuracy_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Accuracy score.

    The formula is as follows:
        accuracy = (1 / N) Σ(i=0 to N-1) I(y_i == t_i),

        where:
            - N - number of samples,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i) - indicator function.
    Args:
        targets: True labels.
        predictions: Predicted class.
    """
    # TODO: Implement computation of accuracy using the formula from the docstring, return the result
    raise NotImplementedError


def precision_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Precision score.

    The formula is as follows:
        precision = TP / (TP + FP),

        where:
            - TP is the number of true positives,
            - FP is the number of false positives.
    Args:
        targets: True labels.
        predictions: Predicted class.
    """
    # TODO: Implement computation of precision score using the formula from the docstring, return the result
    raise NotImplementedError


def recall_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Recall score.

    The formula is as follows:
        recall = TP / (TP + FN),

        where:
            - TP is the number of true positives,
            - FN is the number of false negatives.
    Args:
        targets: True labels.
        predictions: Predicted class.
    """
    # TODO: Implement computation of recall score using the formula from the docstring, return the result
    raise NotImplementedError


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray):
    """Confusion matrix.

    Confusion matrix C with shape KxK:
        c[i, j] - number of observations known to be in class i and predicted to be in class j,

        where:
            - K is the number of classes.

    Args:
        targets: True labels.
        predictions: Predicted class.
    """
    # TODO: Make confusion matrix using the formula from the docstring, return the result
    raise NotImplementedError


def precision_recall_curve(targets: np.ndarray, scores: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """Precision-Recall curve computation for different threshold.

    For every th_n compute precision and recall:
            - P_n = (Σ_i I(t_i == 1) * I(score_i >= th_n)) / (Σ_i I(score_i >= th_n))
            - R_n = (Σ_i I(t_i == 1) * I(score_i >= th_n)) / (Σ_i I(t_i == 1))

    For n from len(thresholds) - 1 to 0 to smooth the curve:
            - P_{n-1} = max(P_n, P_{n-1})

    Args:
        targets: True labels.
        scores: Target scores.

    Returns:
        np.ndarray: Precision values.
        np.ndarray: Recall values.
        np.ndarray: Thresholds.

    Note: The last precision and recall values should be initialized with 0 and 1, respectively,
                and the first ones - with zeros.
    """

    # Sort the entries according to the predicted confidence in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_targets = targets[sorted_indices]

    # Find unique prediction values (out thresholds), their first occurrence, and their frequencies
    thresholds, first_occurrences, prediction_counts = np.unique(sorted_scores, return_index=True, return_counts=True)

    # Reverse the order to start from the highest prediction
    first_occurrences = first_occurrences[::-1]
    prediction_counts = prediction_counts[::-1]

    # Sum labels at the first occurrences of each unique prediction. This will show how many TP came after we change threshold
    new_tp_for_each_threshold = np.add.reduceat(sorted_targets, first_occurrences)

    # Cumulative true positives (TP) and predicted positives (TP + FP)
    tp_cumulative = np.cumsum(new_tp_for_each_threshold)
    predicted_positives_cumulative = np.cumsum(prediction_counts)

    # Number of actual positives (TP + FN)
    number_of_elements_from_p_class = tp_cumulative[-1]

    # Calculate precision and recall
    precision = tp_cumulative / predicted_positives_cumulative
    recall = tp_cumulative / number_of_elements_from_p_class

    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))

    # Smooth curve using maximum possible curve for each segment
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    return precision, recall, thresholds

def average_precision_score(targets: np.ndarray, scores: np.ndarray) -> float:
    """Computes Average Precision metric.

    Average precision is area under precision-recall curve:
        AP = Σ (R_n - R_{n-1}) * P_n,

        where:
            - P_n and R_n are the precision and recall at the n-th threshold
    Args:
        targets: True labels.
        scores: Target scores.
    """

    p, r, _ = precision_recall_curve(targets, scores)
    return np.sum((r[1:] - r[:-1]) * p[:-1])
