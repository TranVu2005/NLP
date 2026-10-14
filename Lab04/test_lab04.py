import math
import unittest

from preprocessing import (
    document_length_statistics,
    has_missing_text,
    normalize_text,
    sigmoid,
    tokenize_text,
    validate_labels,
)


class PreprocessingTests(unittest.TestCase):
    def test_normalization_preserves_negation(self):
        self.assertEqual(normalize_text("NOT good! Don't stop."), "not good don't stop")

    def test_tokenization_preserves_negation_and_contractions(self):
        self.assertEqual(tokenize_text("This is not good; don't go!"), ["this", "is", "not", "good", "don't", "go"])

    def test_missing_text_detection_handles_none_and_blank_strings(self):
        self.assertTrue(has_missing_text(["review", None, " "]))
        self.assertFalse(has_missing_text(["a", "b"]))

    def test_document_length_statistics(self):
        stats = document_length_statistics(["one two", "three", ""])
        self.assertEqual(stats, {"count": 3, "average": 1.0, "median": 1, "min": 0, "max": 2})

    def test_label_validation_rejects_missing_and_unexpected_labels(self):
        validate_labels(["positive", "negative"], {"positive", "negative"})
        with self.assertRaises(ValueError):
            validate_labels(["positive", None], {"positive", "negative"})
        with self.assertRaises(ValueError):
            validate_labels(["positive", "neutral"], {"positive", "negative"})


class NumericalChecks(unittest.TestCase):
    def test_logistic_neuron_and_sigmoid_reference_value(self):
        x, w, b = [1, 2, 0], [0.5, -0.2, 0.4], 0.1
        z = sum(a * v for a, v in zip(x, w)) + b
        self.assertAlmostEqual(z, 0.2)
        self.assertAlmostEqual(sigmoid(z), 0.549834, places=6)

    def test_probability_thresholds(self):
        self.assertEqual(0.72 >= 0.5, True)
        self.assertEqual(0.72 >= 0.8, False)

    def test_binary_confusion_metrics_reference_values(self):
        tp, fn, fp, tn = 80, 20, 10, 90
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(precision, 0.8888889, places=6)
        self.assertAlmostEqual(recall, 0.8)
        self.assertAlmostEqual(f1, 0.8421053, places=6)

    def test_accuracy_trap_has_zero_positive_recall(self):
        accuracy = 950 / 1000
        positive_recall = 0 / 50
        self.assertEqual(accuracy, 0.95)
        self.assertEqual(positive_recall, 0.0)


if __name__ == "__main__":
    unittest.main()
