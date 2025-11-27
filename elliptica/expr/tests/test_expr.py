"""Tests for expression parsing and compilation."""

import numpy as np
import pytest

from elliptica.expr import (
    compile_expression,
    get_variables,
    list_functions,
    list_constants,
    ParseError,
    ValidationError,
    UnknownVariableError,
    LimitExceededError,
)


class TestBasicExpressions:
    """Test basic expression parsing and evaluation."""

    def test_constant(self):
        fn = compile_expression("42")
        assert fn({}) == 42

    def test_float_constant(self):
        fn = compile_expression("3.14159")
        assert fn({}) == pytest.approx(3.14159)

    def test_scientific_notation(self):
        fn = compile_expression("1e-5")
        assert fn({}) == pytest.approx(1e-5)

    def test_variable(self):
        fn = compile_expression("x")
        x = np.array([1, 2, 3])
        result = fn({'x': x})
        np.testing.assert_array_equal(result, x)

    def test_addition(self):
        fn = compile_expression("x + y")
        result = fn({'x': np.array([1, 2]), 'y': np.array([3, 4])})
        np.testing.assert_array_equal(result, [4, 6])

    def test_subtraction(self):
        fn = compile_expression("x - 1")
        result = fn({'x': np.array([5, 10])})
        np.testing.assert_array_equal(result, [4, 9])

    def test_multiplication(self):
        fn = compile_expression("x * 2")
        result = fn({'x': np.array([3, 4])})
        np.testing.assert_array_equal(result, [6, 8])

    def test_division(self):
        fn = compile_expression("x / 2")
        result = fn({'x': np.array([6, 10])})
        np.testing.assert_array_equal(result, [3, 5])

    def test_power(self):
        fn = compile_expression("x ** 2")
        result = fn({'x': np.array([2, 3])})
        np.testing.assert_array_equal(result, [4, 9])

    def test_unary_minus(self):
        fn = compile_expression("-x")
        result = fn({'x': np.array([1, -2])})
        np.testing.assert_array_equal(result, [-1, 2])

    def test_complex_expression(self):
        fn = compile_expression("x * 0.7 + y * 0.3")
        result = fn({'x': np.array([10.0]), 'y': np.array([20.0])})
        np.testing.assert_allclose(result, [13.0])

    def test_parentheses(self):
        fn = compile_expression("(x + y) * 2")
        result = fn({'x': np.array([1.0]), 'y': np.array([2.0])})
        np.testing.assert_allclose(result, [6.0])


class TestBuiltinConstants:
    """Test built-in constants."""

    def test_pi(self):
        fn = compile_expression("pi")
        assert fn({}) == pytest.approx(np.pi)

    def test_e(self):
        fn = compile_expression("e")
        assert fn({}) == pytest.approx(np.e)

    def test_tau(self):
        fn = compile_expression("tau")
        assert fn({}) == pytest.approx(2 * np.pi)

    def test_constant_in_expression(self):
        fn = compile_expression("x * pi")
        result = fn({'x': np.array([1, 2])})
        np.testing.assert_allclose(result, [np.pi, 2 * np.pi])


class TestBuiltinFunctions:
    """Test all built-in functions."""

    # Trigonometric
    def test_sin(self):
        fn = compile_expression("sin(x)")
        x = np.array([0, np.pi / 2, np.pi])
        result = fn({'x': x})
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_cos(self):
        fn = compile_expression("cos(x)")
        result = fn({'x': np.array([0, np.pi])})
        np.testing.assert_allclose(result, [1, -1], atol=1e-10)

    def test_tan(self):
        fn = compile_expression("tan(x)")
        result = fn({'x': np.array([0, np.pi / 4])})
        np.testing.assert_allclose(result, [0, 1], atol=1e-10)

    # Exponential and logarithmic
    def test_exp(self):
        fn = compile_expression("exp(x)")
        result = fn({'x': np.array([0, 1, 2])})
        np.testing.assert_allclose(result, [1, np.e, np.e**2])

    def test_log(self):
        fn = compile_expression("log(x)")
        result = fn({'x': np.array([1, np.e, np.e**2])})
        np.testing.assert_allclose(result, [0, 1, 2])

    def test_log10(self):
        fn = compile_expression("log10(x)")
        result = fn({'x': np.array([1, 10, 100])})
        np.testing.assert_allclose(result, [0, 1, 2])

    # Basic math
    def test_sqrt(self):
        fn = compile_expression("sqrt(x)")
        result = fn({'x': np.array([4, 9, 16])})
        np.testing.assert_array_equal(result, [2, 3, 4])

    def test_abs(self):
        fn = compile_expression("abs(x)")
        result = fn({'x': np.array([-3, 0, 5])})
        np.testing.assert_array_equal(result, [3, 0, 5])

    # Multi-argument functions
    def test_clamp(self):
        fn = compile_expression("clamp(x, 0, 1)")
        result = fn({'x': np.array([-0.5, 0.5, 1.5])})
        np.testing.assert_array_equal(result, [0, 0.5, 1])

    def test_lerp(self):
        fn = compile_expression("lerp(a, b, t)")
        result = fn({
            'a': np.array([0.0]),
            'b': np.array([10.0]),
            't': np.array([0.5])
        })
        np.testing.assert_allclose(result, [5.0])

    def test_lerp_endpoints(self):
        fn = compile_expression("lerp(a, b, t)")
        # t=0 should give a, t=1 should give b
        result_0 = fn({'a': np.array([5.0]), 'b': np.array([15.0]), 't': np.array([0.0])})
        result_1 = fn({'a': np.array([5.0]), 'b': np.array([15.0]), 't': np.array([1.0])})
        np.testing.assert_allclose(result_0, [5.0])
        np.testing.assert_allclose(result_1, [15.0])

    def test_smoothstep(self):
        fn = compile_expression("smoothstep(e0, e1, x)")
        result = fn({
            'e0': np.array([0.0]),
            'e1': np.array([1.0]),
            'x': np.array([0.0, 0.5, 1.0])
        })
        # smoothstep(0,1,0.5) = 0.5^2 * (3 - 2*0.5) = 0.25 * 2 = 0.5
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    # Reductions
    def test_min_reduction(self):
        fn = compile_expression("min(x)")
        result = fn({'x': np.array([5, 2, 8])})
        assert result == 2

    def test_max_reduction(self):
        fn = compile_expression("max(x)")
        result = fn({'x': np.array([5, 2, 8])})
        assert result == 8

    def test_mean_reduction(self):
        fn = compile_expression("mean(x)")
        result = fn({'x': np.array([2, 4, 6])})
        assert result == pytest.approx(4)

    def test_std_reduction(self):
        fn = compile_expression("std(x)")
        result = fn({'x': np.array([1, 2, 3, 4, 5])})
        assert result == pytest.approx(np.std([1, 2, 3, 4, 5]))

    # Global transforms
    def test_normalize(self):
        fn = compile_expression("normalize(x)")
        result = fn({'x': np.array([0, 50, 100])})
        np.testing.assert_allclose(result, [0, 0.5, 1])

    def test_normalize_constant(self):
        """normalize of constant array should return zeros."""
        fn = compile_expression("normalize(x)")
        result = fn({'x': np.array([5, 5, 5])})
        np.testing.assert_allclose(result, [0, 0, 0])

    def test_pclip(self):
        fn = compile_expression("pclip(x, 10, 90)")
        x = np.arange(100, dtype=float)
        result = fn({'x': x})
        # Values should be clipped to 10th and 90th percentile
        assert result.min() >= 9
        assert result.max() <= 90

    def test_clipnorm(self):
        """clipnorm should percentile clip and normalize to [0, 1]."""
        fn = compile_expression("clipnorm(x, 10, 90)")
        x = np.arange(100, dtype=float)
        result = fn({'x': x})
        # Should be normalized to [0, 1]
        assert result.min() == pytest.approx(0, abs=1e-6)
        assert result.max() == pytest.approx(1, abs=1e-6)

    def test_clipnorm_constant_array(self):
        """clipnorm on constant array should return zeros."""
        fn = compile_expression("clipnorm(x, 0.5, 99.5)")
        x = np.array([5.0, 5.0, 5.0])
        result = fn({'x': x})
        np.testing.assert_allclose(result, [0, 0, 0])


class TestGetVariables:
    """Test variable extraction."""

    def test_single_variable(self):
        assert get_variables("x") == {'x'}

    def test_multiple_variables(self):
        assert get_variables("x + y * z") == {'x', 'y', 'z'}

    def test_excludes_constants(self):
        assert get_variables("x * pi + e") == {'x'}

    def test_excludes_functions(self):
        assert get_variables("sin(x) + cos(y)") == {'x', 'y'}

    def test_repeated_variable(self):
        assert get_variables("x + x * x") == {'x'}


class TestErrors:
    """Test error handling."""

    def test_syntax_error(self):
        with pytest.raises(ParseError):
            compile_expression("x +")

    def test_unknown_function(self):
        with pytest.raises(ValidationError, match="Unknown function"):
            compile_expression("unknown(x)")

    def test_wrong_arg_count(self):
        with pytest.raises(ValidationError, match="expects"):
            compile_expression("sin(x, y)")

    def test_missing_variable(self):
        fn = compile_expression("x + y")
        with pytest.raises(UnknownVariableError):
            fn({'x': np.array([1])})

    def test_keyword_args_error_message(self):
        """Keyword arguments should give clear error, not arg count error."""
        with pytest.raises(ValidationError, match="Keyword arguments"):
            compile_expression("sin(x=5)")


class TestSecurityBlocked:
    """Test that dangerous constructs are blocked."""

    def test_blocks_lambda(self):
        with pytest.raises(ValidationError):
            compile_expression("lambda x: x")

    def test_blocks_list_comprehension(self):
        with pytest.raises(ValidationError):
            compile_expression("[x for x in y]")

    def test_blocks_dict_comprehension(self):
        with pytest.raises(ValidationError):
            compile_expression("{k: v for k, v in items}")

    def test_blocks_generator_expression(self):
        with pytest.raises(ValidationError):
            compile_expression("(x for x in y)")

    def test_blocks_attribute_access(self):
        with pytest.raises(ValidationError):
            compile_expression("x.shape")

    def test_blocks_subscript(self):
        with pytest.raises(ValidationError):
            compile_expression("x[0]")

    def test_blocks_slice(self):
        with pytest.raises(ValidationError):
            compile_expression("x[1:5]")

    def test_blocks_comparison(self):
        with pytest.raises(ValidationError):
            compile_expression("x > 5")

    def test_blocks_boolean_and(self):
        with pytest.raises(ValidationError):
            compile_expression("x and y")

    def test_blocks_boolean_or(self):
        with pytest.raises(ValidationError):
            compile_expression("x or y")

    def test_blocks_boolean_not(self):
        with pytest.raises(ValidationError):
            compile_expression("not x")

    def test_blocks_ternary(self):
        with pytest.raises(ValidationError):
            compile_expression("x if y else z")

    def test_blocks_tuple(self):
        with pytest.raises(ValidationError):
            compile_expression("(x, y)")

    def test_blocks_list_literal(self):
        with pytest.raises(ValidationError):
            compile_expression("[1, 2, 3]")

    def test_blocks_dict_literal(self):
        with pytest.raises(ValidationError):
            compile_expression("{'a': 1}")

    def test_blocks_set_literal(self):
        with pytest.raises(ValidationError):
            compile_expression("{1, 2, 3}")

    def test_blocks_string_constant(self):
        with pytest.raises(ValidationError, match="numeric constants"):
            compile_expression("'hello'")

    def test_blocks_true(self):
        with pytest.raises(ValidationError, match="numeric constants"):
            compile_expression("True")

    def test_blocks_false(self):
        with pytest.raises(ValidationError, match="numeric constants"):
            compile_expression("False")

    def test_blocks_none(self):
        with pytest.raises(ValidationError, match="numeric constants"):
            compile_expression("None")

    def test_blocks_method_call(self):
        with pytest.raises(ValidationError):
            compile_expression("x.sum()")

    def test_blocks_import_name(self):
        """__import__ should be blocked as unknown function."""
        with pytest.raises(ValidationError, match="Unknown function"):
            compile_expression("__import__('os')")

    def test_blocks_eval(self):
        with pytest.raises(ValidationError, match="Unknown function"):
            compile_expression("eval('1+1')")

    def test_blocks_exec(self):
        with pytest.raises(ValidationError, match="Unknown function"):
            compile_expression("exec('x=1')")

    def test_blocks_bitwise_or(self):
        with pytest.raises(ValidationError):
            compile_expression("x | y")

    def test_blocks_bitwise_and(self):
        with pytest.raises(ValidationError):
            compile_expression("x & y")

    def test_blocks_floor_division(self):
        with pytest.raises(ValidationError):
            compile_expression("x // y")

    def test_blocks_modulo(self):
        with pytest.raises(ValidationError):
            compile_expression("x % y")


class TestSafetyLimits:
    """Test safety limits."""

    def test_max_nodes_exceeded(self):
        expr = " + ".join(["x"] * 50)
        with pytest.raises(LimitExceededError, match="nodes"):
            compile_expression(expr, max_nodes=10)

    def test_max_depth_exceeded(self):
        # Nested function calls create real AST depth
        expr = "sin(" * 15 + "x" + ")" * 15
        with pytest.raises(LimitExceededError, match="depth"):
            compile_expression(expr, max_depth=10)

    def test_within_default_limits(self):
        # Should succeed with default limits (depth 20, nodes 100)
        # Note: "x + x + x + ..." creates a left-recursive tree with depth = count
        expr = " + ".join(["x"] * 15)  # 15 additions = depth ~16
        fn = compile_expression(expr)
        result = fn({'x': np.array([1.0])})
        assert result[0] == 15.0


class TestEdgeCases:
    """Test edge cases and numeric behavior."""

    def test_empty_string(self):
        with pytest.raises(ParseError):
            compile_expression("")

    def test_whitespace_only(self):
        with pytest.raises(ParseError):
            compile_expression("   ")

    def test_division_by_zero(self):
        fn = compile_expression("x / 0")
        result = fn({'x': np.array([1.0])})
        assert np.isinf(result[0])

    def test_sqrt_negative(self):
        fn = compile_expression("sqrt(x)")
        result = fn({'x': np.array([-1.0])})
        assert np.isnan(result[0])

    def test_log_zero(self):
        fn = compile_expression("log(x)")
        result = fn({'x': np.array([0.0])})
        assert np.isinf(result[0])

    def test_log_negative(self):
        fn = compile_expression("log(x)")
        result = fn({'x': np.array([-1.0])})
        assert np.isnan(result[0])

    def test_reduction_broadcasts_with_array(self):
        """Reductions return scalars that broadcast with arrays."""
        fn = compile_expression("x - min(x)")
        result = fn({'x': np.array([3, 5, 7])})
        np.testing.assert_array_equal(result, [0, 2, 4])


class TestTorchBackend:
    """Test torch tensor support."""

    @pytest.fixture
    def torch(self):
        pytest.importorskip('torch')
        import torch
        return torch

    def test_basic_torch(self, torch):
        fn = compile_expression("x * 2")
        x = torch.tensor([1.0, 2.0, 3.0])
        result = fn({'x': x})
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([2.0, 4.0, 6.0]))

    def test_torch_functions(self, torch):
        fn = compile_expression("sin(x)")
        x = torch.tensor([0.0, np.pi / 2])
        result = fn({'x': x})
        torch.testing.assert_close(
            result, torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0
        )

    def test_torch_numpy_parity(self, torch):
        fn = compile_expression("x * 0.5 + sin(y)")

        x_np = np.array([1.0, 2.0])
        y_np = np.array([0.0, np.pi])
        result_np = fn({'x': x_np, 'y': y_np})

        x_t = torch.tensor(x_np)
        y_t = torch.tensor(y_np)
        result_t = fn({'x': x_t, 'y': y_t})

        np.testing.assert_allclose(result_t.numpy(), result_np, atol=1e-6)

    def test_torch_exp_log(self, torch):
        fn = compile_expression("log(exp(x))")
        x = torch.tensor([1.0, 2.0, 3.0])
        result = fn({'x': x})
        torch.testing.assert_close(result, x, atol=1e-6, rtol=0)

    def test_torch_normalize(self, torch):
        fn = compile_expression("normalize(x)")
        x = torch.tensor([0.0, 50.0, 100.0])
        result = fn({'x': x})
        torch.testing.assert_close(
            result, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6, rtol=0
        )

    def test_torch_mean_with_int_tensor(self, torch):
        """Torch mean should work with integer tensors."""
        fn = compile_expression("mean(x)")
        x = torch.tensor([1, 2, 3, 4, 5])  # int64
        result = fn({'x': x})
        assert result == pytest.approx(3.0)

    def test_torch_std_with_int_tensor(self, torch):
        """Torch std should work with integer tensors."""
        fn = compile_expression("std(x)")
        x = torch.tensor([1, 2, 3, 4, 5])  # int64
        result = fn({'x': x})
        assert result == pytest.approx(torch.tensor([1, 2, 3, 4, 5]).float().std().item())


class TestListHelpers:
    """Test helper functions."""

    def test_list_functions(self):
        funcs = list_functions()
        assert 'sin' in funcs
        assert funcs['sin'] == 1
        assert 'clamp' in funcs
        assert funcs['clamp'] == 3

    def test_list_functions_complete(self):
        """All 18 functions should be listed."""
        funcs = list_functions()
        expected = {
            'sin', 'cos', 'tan', 'sqrt', 'abs', 'exp', 'log', 'log10',
            'clamp', 'lerp', 'smoothstep',
            'min', 'max', 'mean', 'std',
            'normalize', 'pclip', 'clipnorm'
        }
        assert set(funcs.keys()) == expected

    def test_list_constants(self):
        consts = list_constants()
        assert 'pi' in consts
        assert consts['pi'] == pytest.approx(np.pi)

    def test_list_constants_complete(self):
        """All 3 constants should be listed."""
        consts = list_constants()
        assert set(consts.keys()) == {'pi', 'e', 'tau'}
