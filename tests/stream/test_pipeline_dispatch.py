"""Tests for incremental transformer signature dispatch."""

import unittest

import numpy as np

from numcompute_stream.pipeline import _call_partial_fit


class TestPipelineDispatch(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0], [2.0]])
        self.y = np.array([0, 1])

    def test_x_only_method_is_called_once(self):
        class Transformer:
            def __init__(self):
                self.calls = 0

            def partial_fit(self, X):
                self.calls += 1
                self.received_X = X
                return self

        transformer = Transformer()
        result = _call_partial_fit(transformer, self.X, self.y)

        self.assertIs(result, transformer)
        self.assertEqual(transformer.calls, 1)
        self.assertIs(transformer.received_X, self.X)

    def test_named_y_signatures_receive_labels(self):
        class OptionalY:
            def partial_fit(self, X, y=None):
                self.received = (X, y)
                return self

        class RequiredY:
            def partial_fit(self, X, y):
                self.received = (X, y)
                return self

        class KeywordOnlyY:
            def partial_fit(self, X, *, y):
                self.received = (X, y)
                return self

        class PositionalOnlyY:
            def partial_fit(self, X, y, /):
                self.received = (X, y)
                return self

        for component_type in (OptionalY, RequiredY, KeywordOnlyY, PositionalOnlyY):
            with self.subTest(component=component_type.__name__):
                transformer = component_type()
                result = _call_partial_fit(transformer, self.X, self.y)
                self.assertIs(result, transformer)
                self.assertIs(transformer.received[0], self.X)
                self.assertIs(transformer.received[1], self.y)

    def test_keyword_arguments_receive_y(self):
        class Transformer:
            def partial_fit(self, X, **kwargs):
                self.received_X = X
                self.received_kwargs = kwargs
                return self

        transformer = Transformer()
        _call_partial_fit(transformer, self.X, self.y)

        self.assertIs(transformer.received_X, self.X)
        self.assertEqual(set(transformer.received_kwargs), {"y"})
        self.assertIs(transformer.received_kwargs["y"], self.y)

    def test_internal_type_error_is_not_retried_or_replaced(self):
        original_error = TypeError("failure inside transformer")

        class Transformer:
            def __init__(self):
                self.calls = 0

            def partial_fit(self, X, y=None):
                self.calls += 1
                raise original_error

        transformer = Transformer()

        with self.assertRaises(TypeError) as caught:
            _call_partial_fit(transformer, self.X, self.y)

        self.assertIs(caught.exception, original_error)
        self.assertEqual(transformer.calls, 1)

    def test_incompatible_signature_is_rejected_before_execution(self):
        class Transformer:
            def __init__(self):
                self.calls = 0

            def partial_fit(self, X, *, required_setting):
                self.calls += 1
                return self

        transformer = Transformer()

        with self.assertRaisesRegex(TypeError, "Unsupported partial_fit signature"):
            _call_partial_fit(transformer, self.X, self.y)

        self.assertEqual(transformer.calls, 0)

    def test_missing_or_noncallable_method_is_rejected(self):
        class Noncallable:
            partial_fit = None

        for transformer in (object(), Noncallable()):
            with self.subTest(component=type(transformer).__name__):
                with self.assertRaisesRegex(TypeError, "callable partial_fit"):
                    _call_partial_fit(transformer, self.X, self.y)


if __name__ == "__main__":
    unittest.main()