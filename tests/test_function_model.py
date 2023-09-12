from inspect import Parameter, Signature, signature
from sys import version_info
from types import GenericAlias, NoneType
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pytest
from pydantic_core import PydanticCustomError, PydanticKnownError, ValidationError

from pydantic import TypeAdapter
from pydantic.function_model import FunctionModel, get_annotation_type

# Python 3.7 does not support positional-only arguments, which this uses for testing
if version_info <= (3, 7):  # noqa: UP036
    pytest.skip('These tests use positional-only parameters in functions, which are not supported prior to Python 3.8')


def test_fm_init_invalid():
    # These should be rejected, as otherwise it might interfere with internals
    def f_underscore(_param):
        return _param

    def f_underscores(__param):
        return __param

    def f_args(args):
        return args

    def f_kwargs(kwargs):
        return kwargs

    with pytest.raises(PydanticCustomError):
        FunctionModel(f_underscore)
    with pytest.raises(PydanticCustomError):
        FunctionModel(f_underscores)
    with pytest.raises(ValueError):
        FunctionModel(f_args)
    with pytest.raises(ValueError):
        FunctionModel(f_kwargs)


def test_fm_init():
    # Valid models
    models = [
        FunctionModel(func_single_pos),
        FunctionModel(func_single_pos_typed),
        FunctionModel(func_single_pos_defaults),
        FunctionModel(func_single_pos_typed_defaults),
        FunctionModel(func_multiple_pos),
        FunctionModel(func_multiple_pos_typed),
        FunctionModel(func_multiple_pos_defaults),
        FunctionModel(func_multiple_pos_typed_defaults),
        FunctionModel(func_single_args),
        FunctionModel(func_single_args_typed),
        FunctionModel(func_single_poskw),
        FunctionModel(func_single_poskw_typed),
        FunctionModel(func_single_poskw_defaults),
        FunctionModel(func_single_poskw_typed_defaults),
        FunctionModel(func_multiple_poskw),
        FunctionModel(func_multiple_poskw_typed),
        FunctionModel(func_multiple_poskw_defaults),
        FunctionModel(func_multiple_poskw_typed_defaults),
        FunctionModel(func_single_kw),
        FunctionModel(func_single_kw_typed),
        FunctionModel(func_single_kw_defaults),
        FunctionModel(func_single_kw_typed_defaults),
        FunctionModel(func_multiple_kw),
        FunctionModel(func_multiple_kw_typed),
        FunctionModel(func_multiple_kw_defaults),
        FunctionModel(func_multiple_kw_typed_defaults),
        FunctionModel(func_single_kwargs),
        FunctionModel(func_single_kwargs_typed),
        FunctionModel(func_single_pos_args),
        FunctionModel(func_single_pos_args_typed),
        FunctionModel(func_single_pos_args_defaults),
        FunctionModel(func_single_pos_args_typed_defaults),
        FunctionModel(func_multiple_pos_args),
        FunctionModel(func_multiple_pos_args_typed),
        FunctionModel(func_multiple_pos_args_defaults),
        FunctionModel(func_multiple_pos_args_typed_defaults),
        FunctionModel(func_single_pos_poskw),
        FunctionModel(func_single_pos_poskw_typed),
        FunctionModel(func_single_pos_poskw_defaults),
        FunctionModel(func_single_pos_poskw_typed_defaults),
        FunctionModel(func_multiple_pos_poskw),
        FunctionModel(func_multiple_pos_poskw_typed),
        FunctionModel(func_multiple_pos_poskw_defaults),
        FunctionModel(func_multiple_pos_poskw_typed_defaults),
        FunctionModel(func_single_pos_kw),
        FunctionModel(func_single_pos_kw_typed),
        FunctionModel(func_single_pos_kw_defaults),
        FunctionModel(func_single_pos_kw_typed_defaults),
        FunctionModel(func_multiple_pos_kw),
        FunctionModel(func_multiple_pos_kw_typed),
        FunctionModel(func_multiple_pos_kw_defaults),
        FunctionModel(func_multiple_pos_kw_typed_defaults),
        FunctionModel(func_single_pos_kwargs),
        FunctionModel(func_single_pos_kwargs_typed),
        FunctionModel(func_single_pos_kwargs_defaults),
        FunctionModel(func_single_pos_kwargs_typed_defaults),
        FunctionModel(func_multiple_pos_kwargs),
        FunctionModel(func_multiple_pos_kwargs_typed),
        FunctionModel(func_multiple_pos_kwargs_defaults),
        FunctionModel(func_multiple_pos_kwargs_typed_defaults),
        FunctionModel(func_single_args_kw),
        FunctionModel(func_single_args_kw_typed),
        FunctionModel(func_single_args_kw_defaults),
        FunctionModel(func_single_args_kw_typed_defaults),
        FunctionModel(func_multiple_args_kw),
        FunctionModel(func_multiple_args_kw_typed),
        FunctionModel(func_multiple_args_kw_defaults),
        FunctionModel(func_multiple_args_kw_typed_defaults),
        FunctionModel(func_args_kwargs),
        FunctionModel(func_args_kwargs_typed),
        FunctionModel(func_single_poskw_args),
        FunctionModel(func_single_poskw_args_typed),
        FunctionModel(func_single_poskw_args_defaults),
        FunctionModel(func_single_poskw_args_typed_defaults),
        FunctionModel(func_multiple_poskw_args),
        FunctionModel(func_multiple_poskw_args_typed),
        FunctionModel(func_multiple_poskw_args_defaults),
        FunctionModel(func_multiple_poskw_args_typed_defaults),
        FunctionModel(func_single_poskw_kw),
        FunctionModel(func_single_poskw_kw_typed),
        FunctionModel(func_single_poskw_kw_defaults),
        FunctionModel(func_single_poskw_kw_typed_defaults),
        FunctionModel(func_multiple_poskw_kw),
        FunctionModel(func_multiple_poskw_kw_typed),
        FunctionModel(func_multiple_poskw_kw_defaults),
        FunctionModel(func_multiple_poskw_kw_typed_defaults),
        FunctionModel(func_single_poskw_kwargs),
        FunctionModel(func_single_poskw_kwargs_typed),
        FunctionModel(func_single_poskw_kwargs_defaults),
        FunctionModel(func_single_poskw_kwargs_typed_defaults),
        FunctionModel(func_multiple_poskw_kwargs),
        FunctionModel(func_multiple_poskw_kwargs_typed),
        FunctionModel(func_multiple_poskw_kwargs_defaults),
        FunctionModel(func_multiple_poskw_kwargs_typed_defaults),
        FunctionModel(func_single_kw_kwargs),
        FunctionModel(func_single_kw_kwargs_typed),
        FunctionModel(func_single_kw_kwargs_defaults),
        FunctionModel(func_single_kw_kwargs_typed_defaults),
        FunctionModel(func_multiple_kw_kwargs),
        FunctionModel(func_multiple_kw_kwargs_typed),
        FunctionModel(func_multiple_kw_kwargs_defaults),
        FunctionModel(func_multiple_kw_kwargs_typed_defaults),
        FunctionModel(func_single_pos_poskw_args),
        FunctionModel(func_single_pos_poskw_args_typed),
        FunctionModel(func_single_pos_poskw_args_defaults),
        FunctionModel(func_single_pos_poskw_args_typed_defaults),
        FunctionModel(func_multiple_pos_poskw_args),
        FunctionModel(func_multiple_pos_poskw_args_typed),
        FunctionModel(func_multiple_pos_poskw_args_defaults),
        FunctionModel(func_multiple_pos_poskw_args_typed_defaults),
        FunctionModel(func_single_pos_poskw_kw),
        FunctionModel(func_single_pos_poskw_kw_typed),
        FunctionModel(func_single_pos_poskw_kw_defaults),
        FunctionModel(func_single_pos_poskw_kw_typed_defaults),
        FunctionModel(func_multiple_pos_poskw_kw),
        FunctionModel(func_multiple_pos_poskw_kw_typed),
        FunctionModel(func_multiple_pos_poskw_kw_defaults),
        FunctionModel(func_multiple_pos_poskw_kw_typed_defaults),
        FunctionModel(func_single_pos_poskw_kwargs),
        FunctionModel(func_single_pos_poskw_kwargs_typed),
        FunctionModel(func_single_pos_poskw_kwargs_defaults),
        FunctionModel(func_single_pos_poskw_kwargs_typed_defaults),
        FunctionModel(func_multiple_pos_poskw_kwargs),
        FunctionModel(func_multiple_pos_poskw_kwargs_typed),
        FunctionModel(func_multiple_pos_poskw_kwargs_defaults),
        FunctionModel(func_multiple_pos_poskw_kwargs_typed_defaults),
        FunctionModel(func_single_pos_args_kw),
        FunctionModel(func_single_pos_args_kw_typed),
        FunctionModel(func_single_pos_args_kw_defaults),
        FunctionModel(func_single_pos_args_kw_typed_defaults),
        FunctionModel(func_multiple_pos_args_kw),
        FunctionModel(func_multiple_pos_args_kw_typed),
        FunctionModel(func_multiple_pos_args_kw_defaults),
        FunctionModel(func_multiple_pos_args_kw_typed_defaults),
        FunctionModel(func_single_pos_args_kwargs),
        FunctionModel(func_single_pos_args_kwargs_typed),
        FunctionModel(func_single_pos_args_kwargs_defaults),
        FunctionModel(func_single_pos_args_kwargs_typed_defaults),
        FunctionModel(func_multiple_pos_args_kwargs),
        FunctionModel(func_multiple_pos_args_kwargs_typed),
        FunctionModel(func_multiple_pos_args_kwargs_defaults),
        FunctionModel(func_multiple_pos_args_kwargs_typed_defaults),
        FunctionModel(func_single_pos_kw_kwargs),
        FunctionModel(func_single_pos_kw_kwargs_typed),
        FunctionModel(func_single_pos_kw_kwargs_defaults),
        FunctionModel(func_single_pos_kw_kwargs_typed_defaults),
        FunctionModel(func_multiple_pos_kw_kwargs),
        FunctionModel(func_multiple_pos_kw_kwargs_typed),
        FunctionModel(func_multiple_pos_kw_kwargs_defaults),
        FunctionModel(func_multiple_pos_kw_kwargs_typed_defaults),
        FunctionModel(func_single_poskw_args_kw),
        FunctionModel(func_single_poskw_args_kw_typed),
        FunctionModel(func_single_poskw_args_kw_defaults),
        FunctionModel(func_single_poskw_args_kw_typed_defaults),
        FunctionModel(func_multiple_poskw_args_kw),
        FunctionModel(func_multiple_poskw_args_kw_typed),
        FunctionModel(func_multiple_poskw_args_kw_defaults),
        FunctionModel(func_multiple_poskw_args_kw_typed_defaults),
        FunctionModel(func_single_poskw_args_kwargs),
        FunctionModel(func_single_poskw_args_kwargs_typed),
        FunctionModel(func_single_poskw_args_kwargs_defaults),
        FunctionModel(func_single_poskw_args_kwargs_typed_defaults),
        FunctionModel(func_multiple_poskw_args_kwargs),
        FunctionModel(func_multiple_poskw_args_kwargs_typed),
        FunctionModel(func_multiple_poskw_args_kwargs_defaults),
        FunctionModel(func_multiple_poskw_args_kwargs_typed_defaults),
        FunctionModel(func_single_poskw_kw_kwargs),
        FunctionModel(func_single_poskw_kw_kwargs_typed),
        FunctionModel(func_single_poskw_kw_kwargs_defaults),
        FunctionModel(func_single_poskw_kw_kwargs_typed_defaults),
        FunctionModel(func_multiple_poskw_kw_kwargs),
        FunctionModel(func_multiple_poskw_kw_kwargs_typed),
        FunctionModel(func_multiple_poskw_kw_kwargs_defaults),
        FunctionModel(func_multiple_poskw_kw_kwargs_typed_defaults),
        FunctionModel(func_single_args_kw_kwargs),
        FunctionModel(func_single_args_kw_kwargs_typed),
        FunctionModel(func_single_args_kw_kwargs_defaults),
        FunctionModel(func_single_args_kw_kwargs_typed_defaults),
        FunctionModel(func_multiple_args_kw_kwargs),
        FunctionModel(func_multiple_args_kw_kwargs_typed),
        FunctionModel(func_multiple_args_kw_kwargs_defaults),
        FunctionModel(func_multiple_args_kw_kwargs_typed_defaults),
        FunctionModel(func_single_pos_poskw_args_kw),
        FunctionModel(func_single_pos_poskw_args_kw_typed),
        FunctionModel(func_single_pos_poskw_args_kw_defaults),
        FunctionModel(func_single_pos_poskw_args_kw_typed_defaults),
        FunctionModel(func_multiple_pos_poskw_args_kw),
        FunctionModel(func_multiple_pos_poskw_args_kw_typed),
        FunctionModel(func_multiple_pos_poskw_args_kw_defaults),
        FunctionModel(func_multiple_pos_poskw_args_kw_typed_defaults),
        FunctionModel(func_single_pos_poskw_args_kwargs),
        FunctionModel(func_single_pos_poskw_args_kwargs_typed),
        FunctionModel(func_single_pos_poskw_args_kwargs_defaults),
        FunctionModel(func_single_pos_poskw_args_kwargs_typed_defaults),
        FunctionModel(func_multiple_pos_poskw_args_kwargs),
        FunctionModel(func_multiple_pos_poskw_args_kwargs_typed),
        FunctionModel(func_multiple_pos_poskw_args_kwargs_defaults),
        FunctionModel(func_multiple_pos_poskw_args_kwargs_typed_defaults),
        FunctionModel(func_single_pos_poskw_kw_kwargs),
        FunctionModel(func_single_pos_poskw_kw_kwargs_typed),
        FunctionModel(func_single_pos_poskw_kw_kwargs_defaults),
        FunctionModel(func_single_pos_poskw_kw_kwargs_typed_defaults),
        FunctionModel(func_multiple_pos_poskw_kw_kwargs),
        FunctionModel(func_multiple_pos_poskw_kw_kwargs_typed),
        FunctionModel(func_multiple_pos_poskw_kw_kwargs_defaults),
        FunctionModel(func_multiple_pos_poskw_kw_kwargs_typed_defaults),
        FunctionModel(func_single_pos_args_kw_kwargs),
        FunctionModel(func_single_pos_args_kw_kwargs_typed),
        FunctionModel(func_single_pos_args_kw_kwargs_defaults),
        FunctionModel(func_single_pos_args_kw_kwargs_typed_defaults),
        FunctionModel(func_multiple_pos_args_kw_kwargs),
        FunctionModel(func_multiple_pos_args_kw_kwargs_typed),
        FunctionModel(func_multiple_pos_args_kw_kwargs_defaults),
        FunctionModel(func_multiple_pos_args_kw_kwargs_typed_defaults),
        FunctionModel(func_single_poskw_args_kw_kwargs),
        FunctionModel(func_single_poskw_args_kw_kwargs_typed),
        FunctionModel(func_single_poskw_args_kw_kwargs_defaults),
        FunctionModel(func_single_poskw_args_kw_kwargs_typed_defaults),
        FunctionModel(func_multiple_poskw_args_kw_kwargs),
        FunctionModel(func_multiple_poskw_args_kw_kwargs_typed),
        FunctionModel(func_multiple_poskw_args_kw_kwargs_defaults),
        FunctionModel(func_multiple_poskw_args_kw_kwargs_typed_defaults),
        FunctionModel(func_single_pos_poskw_args_kw_kwargs),
        FunctionModel(func_single_pos_poskw_args_kw_kwargs_typed),
        FunctionModel(func_single_pos_poskw_args_kw_kwargs_defaults),
        FunctionModel(func_single_pos_poskw_args_kw_kwargs_typed_defaults),
        FunctionModel(func_multiple_pos_poskw_args_kw_kwargs),
        FunctionModel(func_multiple_pos_poskw_args_kw_kwargs_typed),
        FunctionModel(func_multiple_pos_poskw_args_kw_kwargs_defaults),
        FunctionModel(func_multiple_pos_poskw_args_kw_kwargs_typed_defaults),
    ]

    for model in models:
        assert isinstance(model.function, Callable)  # Model's function is set

        # Check that all the function's parameters are properly represented in the model and metadata
        has_args = False
        has_kwargs = False
        params_count = 0
        kw_count = 0
        poskw_count = 0
        pos_count = 0
        params = model._parameters
        sig = signature(model.function)
        for name, details in sig.parameters.items():
            if details.kind == Parameter.VAR_POSITIONAL:
                has_args = True
                continue
            elif details.kind == Parameter.VAR_KEYWORD:
                has_kwargs = True
                continue
            elif details.kind == Parameter.POSITIONAL_OR_KEYWORD:
                assert params[name]['positional']
                assert params[name]['keyword']
                assert params[name]['set_keyword'] is None
                params_count += 1
                poskw_count += 1
            elif details.kind == Parameter.POSITIONAL_ONLY:
                assert params[name]['positional']
                assert not params[name]['keyword']
                assert params[name]['set_keyword'] is None
                params_count += 1
                pos_count += 1
            elif details.kind == Parameter.KEYWORD_ONLY:
                assert not params[name]['positional']
                assert params[name]['keyword']
                assert params[name]['set_keyword'] is None
                params_count += 1
                kw_count += 1

        assert len(params) == params_count

        model_keys = model._model.model_fields.keys()
        if has_args:
            assert 'args' in model_keys
            args_field = model._model.model_fields.get('args')
            assert args_field.annotation == list
            assert args_field.default is None
            assert len(model_keys) == params_count + 1
        else:
            assert 'args' not in model_keys
            assert len(model_keys) == params_count

        extra = model._model.model_config.get('extra')
        if has_kwargs:
            assert extra == 'allow'
        else:
            assert extra == 'forbid'

        assert isinstance(model._return, TypeAdapter)

        # Check that no arguments are yet defined
        assert model._parsed is None

    # Check that positional-only functions are represented as such in their models
    model = models[4]  # func_multiple_pos
    for param in model._parameters.values():
        assert param['positional']
        assert not param['keyword']

    # Check that keyword-only functions are represented as such in their models
    model = models[23]  # func_multiple_kw_typed
    for param in model._parameters.values():
        assert not param['positional']
        assert param['keyword']

    # Check that positional/keyword functions are represented as such in their models
    model = models[16]  # func_multiple_poskw_defaults
    for param in model._parameters.values():
        assert param['positional']
        assert param['keyword']


def test_fm_update_argument():
    model_pos = FunctionModel(func_single_pos_typed_defaults)
    model_args = FunctionModel(func_multiple_pos_poskw_args_kw_typed_defaults)
    FunctionModel(func_single_kwargs_typed)

    # Regular parameter
    with pytest.raises(AttributeError):
        model_pos.update_argument(0, a_default)  # No arguments yet set, so cannot update them

    model_pos.parse_arguments()
    assert model_pos.get_argument(0) == a_default
    a_new = ((5,), {'five': 6.6})
    model_pos.update_argument(0, a_new)
    assert model_pos.get_argument(0) == a_new
    assert model_pos.get_arguments() == ([a_new], {})

    with pytest.raises(ValidationError):
        model_pos.update_argument(0, None)  # Invalid argument type for this parameter
    with pytest.raises(ValidationError):
        model_pos.update_argument(1, a_default)  # Non-existent parameter and no *args present
    with pytest.raises(ValidationError):
        model_pos.update_argument('args', [a_default])  # Internal-only parameter

    # *args
    model_args.parse_arguments()  # Default means no *args arguments are specified
    assert model_args.get_arguments() == (
        [a_default, b_default],
        {'c': c_default, 'd': d_default, 'e': e_default, 'f': f_default},
    )

    with pytest.raises(AttributeError):
        model_args.update_argument(5, a_default)  # Argument 4 not set, so argument 5 cannot be set, either

    model_args.update_argument(0, a_new)  # Update argument 0
    c_new = {
        'a': {},
        'b': {'b1': {}},
        'c': ((9, 8), {'c1': 7}, {}, ()),
    }
    model_args.update_argument(2, c_new)  # Update argument c/2
    model_args.update_argument(4, d_default)  # Update/add first argument within *args
    assert model_args.get_arguments() == (
        [a_new, b_default, d_default],
        {'c': c_new, 'd': d_default, 'e': e_default, 'f': f_default},
    )
    model_args.update_argument(4, e_default)
    assert model_args.get_arguments() == (
        [a_new, b_default, e_default],
        {'c': c_new, 'd': d_default, 'e': e_default, 'f': f_default},
    )

    with pytest.raises(AttributeError):
        model_args.update_argument(6, a_default)  # Argument 5 not set, so argument 6 cannot be set, either


def test_fm_validate_argument():
    model_pos = FunctionModel(func_single_pos)
    model_kw_typed = FunctionModel(func_single_kw_typed)
    model_poskw = FunctionModel(func_multiple_poskw)
    model_poskw_typed = FunctionModel(func_multiple_poskw_typed)
    model_args = FunctionModel(func_single_args)
    model_kwargs = FunctionModel(func_single_kwargs)

    # POSKW
    orig_params = model_poskw._parameters.items()

    assert model_poskw.validate_argument(0, a_default) == a_default
    assert model_poskw.validate_argument('a', b_default) == b_default
    with pytest.raises(ValidationError):
        model_poskw.validate_argument(4, 5)  # No parameter at pos 4 and *args is not present
    with pytest.raises(ValidationError):
        model_poskw.validate_argument('args', [])  # Forbidden to access 'args' params directly
    with pytest.raises(ValidationError):
        model_poskw.validate_argument('kwargs', {})  # Forbidden to access 'kwargs' params directly
    with pytest.raises(ValidationError):
        model_poskw.validate_argument('blah', 4)  # kw with no kw params or **kwargs present, invalid

    assert model_poskw._parsed is None  # Calls should not have affected model
    assert model_poskw._parameters.items() == orig_params

    # POSKW typed
    orig_params = model_poskw_typed._parameters.items()

    assert model_poskw_typed.validate_argument(1, b_default) == b_default
    with pytest.raises(ValidationError):
        model_poskw_typed.validate_argument(1, a_default)

    assert model_poskw_typed._parsed is None  # Calls should not have affected model
    assert model_poskw_typed._parameters.items() == orig_params

    # POS
    orig_params = model_pos._parameters.items()

    assert model_pos.validate_argument(0, 'blah') == 'blah'
    with pytest.raises(ValidationError):
        model_pos.validate_argument('a', 'blah')

    assert model_pos._parsed is None  # Calls should not have affected model
    assert model_pos._parameters.items() == orig_params

    # KW
    orig_params = model_kw_typed._parameters.items()

    assert model_kw_typed.validate_argument('a', a_default) == a_default
    with pytest.raises(ValidationError):
        model_kw_typed.validate_argument(0, 'blah')

    assert model_kw_typed._parsed is None  # Calls should not have affected model
    assert model_kw_typed._parameters.items() == orig_params

    # *args
    orig_params = model_poskw._parameters.items()

    assert model_args.validate_argument(0, 1)  # Should allow any arbitrary values, as *args is present
    assert model_args.validate_argument(5, 1)  # Should allow any arbitrary values, as *args is present
    with pytest.raises(ValidationError):
        model_args.validate_argument('args', [])  # Forbidden to access 'args' params directly
    with pytest.raises(ValidationError):
        model_poskw.validate_argument('blah', 4)  # kw with no kw params or **kwargs present, invalid

    assert model_poskw._parsed is None  # Calls should not have affected model
    assert model_poskw._parameters.items() == orig_params

    # **kwargs
    orig_params = model_kwargs._parameters.items()

    assert model_kwargs.validate_argument('blah', 5) == 5
    assert model_kwargs.validate_argument('blah', {}) == {}
    with pytest.raises(ValidationError):
        model_kwargs.validate_argument('kwargs', {})  # Forbidden to access 'kwargs' params directly
    with pytest.raises(ValidationError):
        model_kwargs.validate_argument(0, 4)  # No pos args or *args present, invalid

    assert model_kwargs._parsed is None  # Calls should not have affected model
    assert model_kwargs._parameters.items() == orig_params


def test_fm_validate_parse_get_call():
    # Positional, no defaults
    model = FunctionModel(func_multiple_pos_typed)  # `a` is typed `tuple[tuple[int], dict]`

    original_a = ((5,), {'my_arg': 1.0, 'other': 100.1})
    original_b = '15.4'
    original_args = (original_a, original_b)

    with pytest.raises(AttributeError):
        model.get_arguments()
    model.validate_arguments(*original_args)
    with pytest.raises(AttributeError):
        model.get_arguments()
    with pytest.raises(AttributeError):
        model.get_argument(0)

    model.parse_arguments(*original_args)
    first_args = model.get_arguments()
    first_params = model._parameters
    model.parse_arguments(((5,), {'my_arg': 1}), original_b)
    second_args = model.get_arguments()
    second_params = model._parameters
    model.parse_arguments(((5,), {'my_arg': 1, 'other': 100.1}), original_b)

    with pytest.raises(ValidationError):
        model.validate_arguments(*original_args, 2)  # Too many arguments
    with pytest.raises(ValidationError):
        model.parse_arguments()  # Too few arguments
    with pytest.raises(ValidationError):
        model.validate_arguments(original_a)  # Too few arguments
    with pytest.raises(ValidationError):
        model.parse_arguments(original_a, 1)  # Un-parsable second argument for `str` type
    with pytest.raises(ValidationError):
        model.validate_arguments((), original_b)  # Un-parsable first argument for `tuple[tuple[int], dict]` type
    with pytest.raises(ValidationError):
        model.parse_arguments(((), {}), original_b)  # Un-parsable first argument for `tuple[tuple[int], dict]` type
    with pytest.raises(ValidationError):
        model.validate_arguments(
            ((1, 2), {}), original_b
        )  # Un-parsable first argument for `tuple[tuple[int], dict]` type
    with pytest.raises(ValidationError):
        model.validate_arguments(a=a_default, b=b_default)  # Positional-only parameters
    with pytest.raises(ValidationError):
        model.validate_arguments(c=c_default)  # Non-existent parameter and **kwargs is not present

    assert second_args != first_args
    assert second_params == first_params
    assert model.get_arguments() == first_args
    assert model._parameters == first_params

    assert model.validate_arguments(*original_args) == (model._parsed, model._parameters)

    assert model.get_arguments() == ([((5,), {'my_arg': 1.0, 'other': 100.1}), '15.4'], {})
    assert model.get_argument(0) == ((5,), {'my_arg': 1.0, 'other': 100.1})
    assert model.get_argument(1) == '15.4'
    with pytest.raises(PydanticKnownError):
        model.get_argument('a')
    with pytest.raises(PydanticKnownError):
        model.get_argument('b')

    expected = (((5,), {'my_arg': 1.0, 'other': 100.1}), '15.4')
    assert model.call() == expected
    assert model.call(func_multiple_pos_typed_defaults) == expected
    with pytest.raises(TypeError):
        model.call(func_single_pos)

    # Keyword, defaults
    model = FunctionModel(func_multiple_kw_typed_defaults)

    val_a = ((5,), {})
    model.parse_arguments(a=val_a)
    assert model.get_arguments() == ([], {'a': val_a, 'b': b_default})
    assert model.get_argument('a') == val_a
    assert model.get_argument('b') == b_default

    with pytest.raises(PydanticKnownError):
        model.get_argument(0)
    with pytest.raises(PydanticKnownError):
        model.get_argument(1)

    model.parse_arguments()  # Use only defaults

    with pytest.raises(ValidationError):
        model.validate_arguments(a_default)  # No positional parameters or *args, so invalid

    assert (
        model.validate_arguments()
        == (model._parsed, model._parameters)
        == model.validate_arguments(a=a_default, b=b_default)
        == model.validate_arguments(b=b_default)
        == model.validate_arguments(a=a_default)
    )
    assert model.get_arguments() == ([], {'a': a_default, 'b': b_default})
    assert model.get_argument('a') == a_default
    assert model.get_argument('b') == b_default

    assert model.call() == (a_default, b_default)

    # Positional/Keyword, not type hinted
    model = FunctionModel(func_multiple_poskw_defaults)

    model.parse_arguments(a_default, b=b_default)
    model.parse_arguments(a_default, b_default)
    model.parse_arguments(a=a_default, b=b_default)
    model.parse_arguments(b_default, a_default)
    model.parse_arguments()

    with pytest.raises(ValidationError):
        model.parse_arguments(a_default, a=a_default)  # Setting same argument twice
    with pytest.raises(ValidationError):
        model.parse_arguments(a_default, b_default, a=a_default)  # Setting same argument twice
    with pytest.raises(ValidationError):
        model.parse_arguments(a_default, b_default, b=b_default)  # Setting same argument twice

    parsed1, metadata1 = model.validate_arguments()
    parsed2, metadata2 = model.validate_arguments(a_default)
    parsed3, metadata3 = model.validate_arguments(b=b_default)
    parsed4, metadata4 = model.validate_arguments(a_default, b_default)
    parsed5, metadata5 = model.validate_arguments(a_default, b=b_default)
    parsed6, metadata6 = model.validate_arguments(b=b_default, a=a_default)

    assert parsed1 == parsed2 == parsed3 == parsed4 == parsed5 == parsed6
    assert metadata1 == metadata3 == metadata6
    assert metadata2 == metadata5
    assert metadata1 != metadata2

    # All
    model = FunctionModel(func_single_pos_poskw_args_kw_kwargs_typed_defaults)

    with pytest.raises(ValidationError):
        model.validate_arguments(_my_kw=5)
    with pytest.raises(ValidationError):
        model.validate_arguments(args=[1, 2])
    with pytest.raises(ValidationError):
        model.validate_arguments(kwargs={'a': a_default, 'b': b_default})

    model.parse_arguments(my_kw='blah')
    assert model.get_arguments() == ([a_default], {'c': c_default, 'b': b_default, 'my_kw': 'blah'})
    assert model.get_argument('my_kw') == 'blah'
    with pytest.raises(AttributeError):
        model.get_argument(3)
    with pytest.raises(PydanticCustomError):
        model.get_argument('_my_kw')
    with pytest.raises(PydanticCustomError):
        model.get_argument('args')
    with pytest.raises(PydanticCustomError):
        model.get_argument('kwargs')

    model.parse_arguments(a_default, b_default, 3, 4, c=c_default)
    assert model.get_arguments() == ([a_default, b_default, 3, 4], {'c': c_default})
    assert model.get_argument(2) == 3
    assert model.get_argument(3) == 4
    with pytest.raises(AttributeError):
        model.get_argument(4)
    with pytest.raises(PydanticCustomError):
        model.get_argument('args')
    with pytest.raises(PydanticCustomError):
        model.get_argument('kwargs')


def test_fm_call():
    ...


def test_fm_get_parameter():
    # Positional-only
    model = FunctionModel(func_single_pos)

    assert model._get_parameter(0) == ('a', {'positional': True, 'keyword': False, 'set_keyword': None})

    with pytest.raises(PydanticKnownError):
        model._get_parameter('a')  # Positional-only
    with pytest.raises(PydanticKnownError):
        model._get_parameter('blah')  # Unspecified keyword parameter with no **kwargs
    with pytest.raises(PydanticKnownError):
        model._get_parameter(1)  # Unspecified positional parameter with no *args
    with pytest.raises(PydanticCustomError):
        model._get_parameter('args')  # Forbidden keyword
    with pytest.raises(PydanticCustomError):
        model._get_parameter('kwargs')  # Forbidden keyword

    # Keyword-only
    model = FunctionModel(func_single_kw)

    assert model._get_parameter('a') == ('a', {'positional': False, 'keyword': True, 'set_keyword': None})

    with pytest.raises(PydanticKnownError):
        model._get_parameter(0)  # Keyword-only
    with pytest.raises(PydanticKnownError):
        model._get_parameter('blah')  # Unspecified keyword parameter with no **kwargs
    with pytest.raises(PydanticKnownError):
        model._get_parameter(1)  # Unspecified positional parameter with no *args
    with pytest.raises(PydanticCustomError):
        model._get_parameter('args')  # Forbidden keyword
    with pytest.raises(PydanticCustomError):
        model._get_parameter('kwargs')  # Forbidden keyword

    # Positional/Keyword
    model = FunctionModel(func_single_poskw)

    poskw_a = ('a', {'positional': True, 'keyword': True, 'set_keyword': None})
    assert model._get_parameter(0) == poskw_a
    assert model._get_parameter('a') == poskw_a

    with pytest.raises(PydanticKnownError):
        model._get_parameter('blah')  # Unspecified keyword parameter with no **kwargs
    with pytest.raises(PydanticKnownError):
        model._get_parameter(1)  # Unspecified positional parameter with no *args
    with pytest.raises(PydanticCustomError):
        model._get_parameter('args')  # Forbidden keyword
    with pytest.raises(PydanticCustomError):
        model._get_parameter('kwargs')  # Forbidden keyword

    # All
    model = FunctionModel(func_multiple_pos_poskw_args_kw_kwargs_typed_defaults)

    assert model._get_parameter(0) == ('a', {'positional': True, 'keyword': False, 'set_keyword': None})
    with pytest.raises(PydanticKnownError):
        model._get_parameter('a')  # Positional-only

    poskw_c = ('c', {'positional': True, 'keyword': True, 'set_keyword': None})
    assert model._get_parameter(2) == poskw_c  # Positional/keyword
    assert model._get_parameter('c') == poskw_c

    assert model._get_parameter('e') == ('e', {'positional': False, 'keyword': True, 'set_keyword': None})

    args = ('args', {'positional': True, 'keyword': False})
    kwargs = ('kwargs', {'positional': False, 'keyword': True})
    assert model._get_parameter(4) == args  # Within *args
    assert model._get_parameter(50) == args
    assert model._get_parameter('g') == kwargs  # Within **kwargs
    assert model._get_parameter('blah') == kwargs

    with pytest.raises(PydanticCustomError):
        model._get_parameter('args')  # Forbidden keyword
    with pytest.raises(PydanticCustomError):
        model._get_parameter('kwargs')  # Forbidden keyword


def test_get_annotation_type():
    # No type hints, as gathered by the `inspect` module
    assert get_annotation_type(Parameter.empty) == Any
    assert get_annotation_type(Signature.empty) == Any

    # `None` variants
    assert get_annotation_type(None) == NoneType
    assert get_annotation_type(NoneType) == NoneType

    # Tuple variants
    assert get_annotation_type(tuple) == tuple
    assert get_annotation_type((int, float)) == Tuple[int, float]
    assert get_annotation_type((int, ...)) == Tuple[int, ...]
    assert get_annotation_type(tuple[int, float]) == tuple[int, float]
    assert get_annotation_type(((int, float), str, (int, float))) == Tuple[Tuple[int, float], str, Tuple[int, float]]
    assert get_annotation_type(Tuple) == Tuple
    assert get_annotation_type(Tuple[int, float]) == Tuple[int, float]
    assert get_annotation_type(Tuple[tuple[int, float], str]) == Tuple[tuple[int, float], str]

    # Incompatible types
    assert get_annotation_type(MyClass) == Any
    with pytest.raises(PydanticCustomError):
        get_annotation_type(MyClass, strict=True)


# Functions to test against


class MyClass:
    """Example class. Used for checking compatibility with arbitrary classes."""

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


a_default = ((1,), {'two': 3.3})
b_default = 'blah'
c_default = {'my_dict': {}, 'my_other_dict': {'blah': ()}, 'my_tuple': ((1, 2), {'blah': 3}, {}, ())}
d_default = MyClass(a_default, b_default, c_default)
e_default = tuple
f_default = None

a_types = [
    tuple[tuple[int], dict[str, float]],
    tuple[tuple, dict[str, float]],
    tuple[tuple[int], dict],
    tuple[tuple, dict],
    tuple,
    (tuple[int], dict[str, float]),
    (tuple, dict[str, float]),
    (tuple[int], dict),
    (tuple, dict),
    ((int,), dict),
    Tuple[tuple[int], dict[str, float]],
    Tuple[tuple, dict[str, float]],
    Tuple[tuple[int], dict],
    Tuple[tuple, dict],
    Tuple,
    Tuple[Tuple[int], dict[str, float]],
    Tuple[Tuple, dict[str, float]],
    Tuple[Tuple[int], dict],
    Tuple[Tuple, dict],
    Tuple[tuple[int], Dict[str, float]],
    Tuple[tuple, Dict[str, float]],
    Tuple[tuple[int], Dict],
    Tuple[tuple, Dict],
    tuple[Tuple[int], dict[str, float]],
    tuple[Tuple, dict[str, float]],
    tuple[Tuple[int], dict],
    tuple[Tuple, dict],
    tuple[tuple[int], Dict[str, float]],
    tuple[tuple, Dict[str, float]],
    tuple[tuple[int], Dict],
    tuple[tuple, Dict],
    (Tuple[int], Dict[str, float]),
    (Tuple, Dict),
]
b_type = str
c_types = [
    dict[str, Any],
    Dict[str, Any],
    dict,
    Dict,
]
d_type = MyClass
e_types = [
    Union[None, GenericAlias],
    Optional[GenericAlias],
]
f_types = [
    Any,
    None,
    NoneType,
]
args_types = [
    tuple,
    Tuple,
]
kwargs_types = [
    Dict,
    dict,
]


def func_single_pos(a, /):
    return a


def func_single_pos_typed(a: a_types[0], /) -> a_types[0]:
    return a


def func_single_pos_defaults(a=a_default, /):
    return a


def func_single_pos_typed_defaults(a: a_types[1] = a_default, /) -> a_types[1]:
    return a


def func_multiple_pos(a, b, c, d, /):
    return a, b, c, d


def func_multiple_pos_typed(a: a_types[2], b: b_type, /) -> (a_types[2], b_type):
    return a, b


def func_multiple_pos_defaults(a=a_default, b=b_default, /):
    return a, b


def func_multiple_pos_typed_defaults(a: a_types[3] = a_default, b: b_type = b_default, /) -> (a_types[3], b_type):
    return a, b


def func_single_poskw(a):
    return a


def func_single_poskw_typed(a: a_types[4]) -> a_types[4]:
    return a


def func_single_poskw_defaults(a=a_default):
    return a


def func_single_poskw_typed_defaults(a: a_types[5] = a_default) -> a_types[5]:
    return a


def func_multiple_poskw(a, b):
    return a, b


def func_multiple_poskw_typed(a: a_types[6], b: b_type) -> tuple[a_types[6], b_type]:
    return a, b


def func_multiple_poskw_defaults(a=a_default, b=b_default):
    return a, b


def func_multiple_poskw_typed_defaults(a: a_types[7] = a_default, b: b_type = b_default) -> tuple[a_types[7], b_type]:
    return a, b


def func_single_poskw_args(a, *args):
    return a, args


def func_single_poskw_args_typed(a: a_types[8], *args) -> (a_types[8], args_types[0]):
    return a, args


def func_single_poskw_args_defaults(a=a_default, *args):
    return a, args


def func_single_poskw_args_typed_defaults(a: a_types[9] = a_default, *args) -> (a_types[9], args_types[1]):
    return a, args


def func_multiple_poskw_args(a, b, *args):
    return a, b, args


def func_multiple_poskw_args_typed(a: a_types[10], b: b_type, *args) -> (a_types[10], b_type, args_types[0]):
    return a, b, args


def func_multiple_poskw_args_defaults(a=a_default, b=b_default, *args):
    return a, b, args


def func_multiple_poskw_args_typed_defaults(
    a: a_types[11] = a_default, b: b_type = b_default, *args
) -> (a_types[11], b_type, args_types[1]):
    return a, b, args


def func_single_kw(*, a):
    return a


def func_single_kw_typed(*, a: a_types[12]) -> a_types[12]:
    return a


def func_single_kw_defaults(*, a=a_default):
    return a


def func_single_kw_typed_defaults(*, a: a_types[13] = a_default) -> a_types[13]:
    return a


def func_multiple_kw(*, a, b):
    return a, b


def func_multiple_kw_typed(*, a: a_types[14], b: b_type) -> tuple[a_types[14], b_type]:
    return a, b


def func_multiple_kw_defaults(*, a=a_default, b=b_default):
    return a, b


def func_multiple_kw_typed_defaults(*, a: a_types[15] = a_default, b: b_type = b_default) -> tuple[a_types[15], b_type]:
    return a, b


def func_single_args(*args):
    return args


def func_single_args_typed(*args) -> args_types[0]:
    return args


def func_single_kwargs(**kwargs):
    return kwargs


def func_single_kwargs_typed(**kwargs) -> kwargs_types[0]:
    return kwargs


def func_single_pos_poskw(a, /, b):
    return a, b


def func_single_pos_poskw_typed(a: a_types[16], /, b: b_type) -> (a_types[16], b_type):
    return a, b


def func_single_pos_poskw_defaults(a=a_default, /, b=b_default):
    return a, b


def func_single_pos_poskw_typed_defaults(a: a_types[17] = a_default, /, b: b_type = b_default) -> (a_types[17], b_type):
    return a, b


def func_multiple_pos_poskw(a, b, /, c, d):
    return a, b, c, d


def func_multiple_pos_poskw_typed(
    a: a_types[18], b: b_type, /, c: c_types[0], d: d_type
) -> (a_types[18], b_type, c_types[0], d_type):
    return a, b, c, d


def func_multiple_pos_poskw_defaults(a=a_default, b=b_default, /, c=c_default, d=d_default):
    return a, b, c, d


def func_multiple_pos_poskw_typed_defaults(
    a: a_types[19] = a_default, b: b_type = b_default, /, c: c_types[1] = c_default, d: d_type = d_default
) -> (a_types[19], b_type, c_types[1], d_type):
    return a, b, c, d


def func_single_pos_args(a, /, *args):
    return a, args


def func_single_pos_args_typed(a: a_types[20], /, *args) -> tuple[a_types[20], args_types[1]]:
    return a, args


def func_single_pos_args_defaults(a=a_default, /, *args):
    return a, args


def func_single_pos_args_typed_defaults(a: a_types[21] = a_default, /, *args) -> tuple[a_types[21], args_types[0]]:
    return a, args


def func_multiple_pos_args(a, b, /, *args):
    return a, b, args


def func_multiple_pos_args_typed(a: a_types[22], b: b_type, /, *args) -> tuple[a_types[22], b_type, args_types[1]]:
    return a, b, args


def func_multiple_pos_args_defaults(a=a_default, b=b_default, /, *args):
    return a, b, args


def func_multiple_pos_args_typed_defaults(
    a: a_types[23] = a_default, b: b_type = b_default, /, *args
) -> tuple[a_types[23], b_type, args_types[0]]:
    return a, b, args


def func_single_pos_kw(a, /, *, b):
    return a, b


def func_single_pos_kw_typed(a: a_types[24], /, *, b: b_type) -> (a_types[24], b_type):
    return a, b


def func_single_pos_kw_defaults(a=a_default, /, *, b=b_default):
    return a, b


def func_single_pos_kw_typed_defaults(a: a_types[25] = a_default, /, *, b: b_type = b_default) -> (a_types[25], b_type):
    return a, b


def func_multiple_pos_kw(a, b, /, *, c, d):
    return a, b, c, d


def func_multiple_pos_kw_typed(
    a: a_types[26], b: b_type, /, *, c: c_types[2], d: d_type
) -> (a_types[26], b_type, c_types[2], d_type):
    return a, b, c, d


def func_multiple_pos_kw_defaults(a=a_default, b=b_default, /, *, c=c_default, d=d_default):
    return a, b, c, d


def func_multiple_pos_kw_typed_defaults(
    a: a_types[27] = a_default, b: b_type = b_default, /, *, c: c_types[3] = c_default, d: d_type = d_default
) -> (a_types[27], b_type, c_types[3], d_type):
    return a, b, c, d


def func_single_pos_kwargs(a, /, **kwargs):
    return a, kwargs


def func_single_pos_kwargs_typed(a: a_types[28], /, **kwargs) -> tuple[a_types[28], kwargs_types[1]]:
    return a, kwargs


def func_single_pos_kwargs_defaults(a=a_default, /, **kwargs):
    return a, kwargs


def func_single_pos_kwargs_typed_defaults(
    a: a_types[29] = a_default, /, **kwargs
) -> tuple[a_types[29], kwargs_types[0]]:
    return a, kwargs


def func_multiple_pos_kwargs(a, b, /, **kwargs):
    return a, b, kwargs


def func_multiple_pos_kwargs_typed(
    a: a_types[30], b: b_type, /, **kwargs
) -> tuple[a_types[30], b_type, kwargs_types[1]]:
    return a, b, kwargs


def func_multiple_pos_kwargs_defaults(a=a_default, b=b_default, /, **kwargs):
    return a, b, kwargs


def func_multiple_pos_kwargs_typed_defaults(
    a: a_types[31] = a_default, b: b_type = b_default, /, **kwargs
) -> tuple[a_types[31], b_type, kwargs_types[0]]:
    return a, b, kwargs


def func_single_args_kw(*args, a):
    return args, a


def func_single_args_kw_typed(*args, a: a_types[0]) -> (args_types[1], a_types[0]):
    return args, a


def func_single_args_kw_defaults(*args, a=a_default):
    return args, a


def func_single_args_kw_typed_defaults(*args, a: a_types[1] = a_default) -> (args_types[0], a_types[1]):
    return args, a


def func_multiple_args_kw(*args, a, b):
    return args, a, b


def func_multiple_args_kw_typed(*args, a: a_types[2], b: b_type) -> (args_types[1], a_types[2], b_type):
    return args, a, b


def func_multiple_args_kw_defaults(*args, a=a_default, b=b_default):
    return args, a, b


def func_multiple_args_kw_typed_defaults(
    *args, a: a_types[3] = a_default, b: b_type = b_default
) -> (args_types[0], a_types[3], b_type):
    return args, a, b


def func_args_kwargs(*args, **kwargs):
    return args, kwargs


def func_args_kwargs_typed(*args, **kwargs) -> tuple[args_types[1], kwargs_types[1]]:
    return args, kwargs


def func_single_poskw_kw(a, *, b):
    return a, b


def func_single_poskw_kw_typed(a: a_types[4], *, b: b_type) -> (a_types[4], b_type):  # TODO: FROM HERE
    return a, b


def func_single_poskw_kw_defaults(a=a_default, *, b=b_default):
    return a, b


def func_single_poskw_kw_typed_defaults(
    a: a_types[5] = a_default, *, b: b_type = b_default
) -> (tuple[tuple, dict], str):
    return a, b


def func_multiple_poskw_kw(a, b, *, c, d):
    return a, b, c, d


def func_multiple_poskw_kw_typed(
    a: a_types[6], b: b_type, *, c: c_types[0], d: d_type
) -> tuple[(tuple, dict), str, dict, MyClass]:
    return a, b, c, d


def func_multiple_poskw_kw_defaults(a=a_default, b=b_default, *, c=c_default, d=d_default):
    return a, b, c, d


def func_multiple_poskw_kw_typed_defaults(
    a: a_types[7] = a_default, b: b_type = b_default, *, c: c_types[1] = c_default, d: d_type = d_default
) -> ((tuple, dict), str, dict, dict):
    return a, b, c, d


def func_single_kw_kwargs(*, a, **kwargs):
    return a, kwargs


def func_single_kw_kwargs_typed(*, a: a_types[8], **kwargs) -> ((tuple, dict), dict):
    return a, kwargs


def func_single_kw_kwargs_defaults(*, a=a_default, **kwargs):
    return a, kwargs


def func_single_kw_kwargs_typed_defaults(*, a: a_types[9] = a_default, **kwargs) -> ((tuple, dict), dict):
    return a, kwargs


def func_multiple_kw_kwargs(*, a, b, **kwargs):
    return a, b, kwargs


def func_multiple_kw_kwargs_typed(*, a: a_types[10], b: b_type, **kwargs) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, kwargs


def func_multiple_kw_kwargs_defaults(*, a=a_default, b=b_default, **kwargs):
    return a, b, kwargs


def func_multiple_kw_kwargs_typed_defaults(
    *, a: a_types[11] = a_default, b: b_type = b_default, **kwargs
) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, kwargs


def func_single_poskw_kwargs(a, **kwargs):
    return a, kwargs


def func_single_poskw_kwargs_typed(a: a_types[12], **kwargs) -> ((tuple, dict), dict):
    return a, kwargs


def func_single_poskw_kwargs_defaults(a=a_default, **kwargs):
    return a, kwargs


def func_single_poskw_kwargs_typed_defaults(a: a_types[13] = a_default, **kwargs) -> ((tuple, dict), dict):
    return a, kwargs


def func_multiple_poskw_kwargs(a, b, **kwargs):
    return a, b, kwargs


def func_multiple_poskw_kwargs_typed(a: a_types[14], b: b_type, **kwargs) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, kwargs


def func_multiple_poskw_kwargs_defaults(a=a_default, b=b_default, **kwargs):
    return a, b, kwargs


def func_multiple_poskw_kwargs_typed_defaults(
    a: a_types[15] = a_default, b: b_type = b_default, **kwargs
) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, kwargs


def func_single_pos_poskw_args(a, /, b, *args):
    return a, b, args


def func_single_pos_poskw_args_typed(a: a_types[16], /, b: b_type, *args) -> ((tuple, dict), str, tuple):
    return a, b, args


def func_single_pos_poskw_args_defaults(a=a_default, /, b=b_default, *args):
    return a, b, args


def func_single_pos_poskw_args_typed_defaults(
    a: a_types[17] = a_default, /, b: b_type = b_default, *args
) -> ((tuple, dict), str, tuple):
    return a, b, args


def func_multiple_pos_poskw_args(a, b, /, c, d, *args):
    return a, b, c, d, args


def func_multiple_pos_poskw_args_typed(
    a: a_types[18], b: b_type, /, c: c_types[2], d: d_type, *args
) -> ((tuple, dict), str, dict, MyClass, tuple):
    return a, b, c, d, args


def func_multiple_pos_poskw_args_defaults(a=a_default, b=b_default, /, c=c_default, d=d_default, *args):
    return a, b, c, d, args


def func_multiple_pos_poskw_args_typed_defaults(
    a: a_types[19] = a_default, b: b_type = b_default, /, c: c_types[3] = c_default, d: d_type = d_default, *args
) -> ((tuple, dict), str, dict, MyClass):
    return a, b, c, d, args


def func_single_pos_poskw_kw(a, /, b, *, c):
    return a, b, c


def func_single_pos_poskw_kw_typed(a: a_types[20], /, b: b_type, *, c: c_types[0]) -> ((tuple, dict), str, dict):
    return a, b, c


def func_single_pos_poskw_kw_defaults(a=a_default, /, b=b_default, *, c=c_default):
    return a, b, c


def func_single_pos_poskw_kw_typed_defaults(
    a: a_types[21] = a_default, /, b: b_type = b_default, *, c: c_types[1] = c_default
) -> ((tuple, dict), str, dict):
    return a, b, c


def func_multiple_pos_poskw_kw(a, b, /, c, d, *, e, f):
    return a, b, c, d, e, f


def func_multiple_pos_poskw_kw_typed(
    a: a_types[22], b: b_type, /, c: c_types[2], d: d_type, *, e: e_types[0], f: f_types[0]
) -> ((tuple, dict), str, dict, MyClass):
    return a, b, c, d, e, f


def func_multiple_pos_poskw_kw_defaults(
    a=a_default, b=b_default, /, c=c_default, d=d_default, *, e=e_default, f=f_default
):
    return a, b, c, d, e, f


def func_multiple_pos_poskw_kw_typed_defaults(
    a: a_types[23] = a_default,
    b: b_type = b_default,
    /,
    c: c_types[3] = c_default,
    d: d_type = d_default,
    *,
    e: e_types[1],
    f: f_types[1],
) -> ((tuple, dict), str, dict, MyClass):
    return a, b, c, d, e, f


def func_single_pos_poskw_kwargs(a, /, b, **kwargs):
    return a, b, kwargs


def func_single_pos_poskw_kwargs_typed(a: a_types[24], /, b: b_type, **kwargs) -> ((tuple, dict), str, dict):
    return a, b, kwargs


def func_single_pos_poskw_kwargs_defaults(a=a_default, /, b=b_default, **kwargs):
    return a, b, kwargs


def func_single_pos_poskw_kwargs_typed_defaults(
    a: a_types[25] = a_default, /, b: b_type = b_default, **kwargs
) -> (tuple, str, dict):
    return a, b, kwargs


def func_multiple_pos_poskw_kwargs(a, b, /, c, d, **kwargs):
    return a, b, c, d, kwargs


def func_multiple_pos_poskw_kwargs_typed(
    a: a_types[26], b: b_type, /, c: c_types[0], d: d_type, **kwargs
) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, c, d, kwargs


def func_multiple_pos_poskw_kwargs_defaults(a=a_default, b=b_default, /, c=c_default, d=d_default, **kwargs):
    return a, b, c, d, kwargs


def func_multiple_pos_poskw_kwargs_typed_defaults(
    a: a_types[27] = a_default, b: b_type = b_default, /, c: c_types[1] = c_default, d: d_type = d_default, **kwargs
) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, c, d, kwargs


def func_single_pos_args_kw(a, /, *args, b):
    return a, args, b


def func_single_pos_args_kw_typed(a: a_types[28], /, *args, b: b_type) -> ((tuple, dict), tuple, str):
    return a, args, b


def func_single_pos_args_kw_defaults(a=a_default, /, *args, b=b_default):
    return a, args, b


def func_single_pos_args_kw_typed_defaults(
    a: a_types[29] = a_default, /, *args, b: b_type = b_default
) -> ((tuple, dict), tuple, str):
    return a, args, b


def func_multiple_pos_args_kw(a, b, /, *args, c, d):
    return a, b, args, c, d


def func_multiple_pos_args_kw_typed(
    a: a_types[30], b: b_type, /, *args, c: c_types[2], d: d_type
) -> ((tuple, dict), str, tuple, dict, MyClass):
    return a, b, args, c, d


def func_multiple_pos_args_kw_defaults(a=a_default, b=b_default, /, *args, c=c_default, d=d_default):
    return a, b, args, c, d


def func_multiple_pos_args_kw_typed_defaults(
    a: a_types[31] = a_default, b: b_type = b_default, /, *args, c: c_types[3] = c_default, d: d_type = d_default
) -> ((tuple, dict), str, tuple, dict, MyClass):
    return a, b, args, c, d


def func_single_pos_args_kwargs(a, /, *args, **kwargs):
    return a, args, kwargs


def func_single_pos_args_kwargs_typed(a: a_types[0], /, *args, **kwargs) -> ((tuple, dict), tuple, dict):
    return a, args, kwargs


def func_single_pos_args_kwargs_defaults(a=a_default, /, *args, **kwargs):
    return a, args, kwargs


def func_single_pos_args_kwargs_typed_defaults(
    a: a_types[1] = a_default, /, *args, **kwargs
) -> ((tuple, dict), tuple, dict):
    return a, args, kwargs


def func_multiple_pos_args_kwargs(a, b, /, *args, **kwargs):
    return a, b, args, kwargs


def func_multiple_pos_args_kwargs_typed(
    a: a_types[2], b: b_type, /, *args, **kwargs
) -> ((tuple, dict), str, dict, MyClass, tuple, dict):
    return a, b, args, kwargs


def func_multiple_pos_args_kwargs_defaults(a=a_default, b=b_default, /, *args, **kwargs):
    return a, b, args, kwargs


def func_multiple_pos_args_kwargs_typed_defaults(
    a: a_types[3] = a_default, b: b_type = b_default, /, *args, **kwargs
) -> ((tuple, dict), str, dict, MyClass, tuple, dict):
    return a, b, args, kwargs


def func_single_pos_kw_kwargs(a, /, *, b, **kwargs):
    return a, b, kwargs


def func_single_pos_kw_kwargs_typed(a: a_types[4], /, *, b: b_type, **kwargs) -> ((tuple, dict), str, dict):
    return a, b, kwargs


def func_single_pos_kw_kwargs_defaults(a=a_default, /, *, b=b_default, **kwargs):
    return a, b, kwargs


def func_single_pos_kw_kwargs_typed_defaults(
    a: a_types[5] = a_default, /, *, b: b_type = b_default, **kwargs
) -> ((tuple, dict), str, dict):
    return a, b, kwargs


def func_multiple_pos_kw_kwargs(a, b, /, *, c, d, **kwargs):
    return a, b, c, d, kwargs


def func_multiple_pos_kw_kwargs_typed(
    a: a_types[6], b: b_type, /, *, c: c_types[0], d: d_type, **kwargs
) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, c, d, kwargs


def func_multiple_pos_kw_kwargs_defaults(a=a_default, b=b_default, /, *, c=c_default, d=d_default, **kwargs):
    return a, b, c, d, kwargs


def func_multiple_pos_kw_kwargs_typed_defaults(
    a: a_types[7] = a_default, b: b_type = b_default, /, *, c: c_types[1] = c_default, d: d_type = d_default, **kwargs
) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, c, d, kwargs


def func_single_poskw_args_kw(a, *args, b):
    return a, args, b


def func_single_poskw_args_kw_typed(a: a_types[8], *args, b: b_type) -> ((tuple, dict), tuple, str):
    return a, args, b


def func_single_poskw_args_kw_defaults(a=a_default, *args, b=b_default):
    return a, args, b


def func_single_poskw_args_kw_typed_defaults(
    a: a_types[9] = a_default, *args, b: b_type = b_default
) -> ((tuple, dict), tuple, str):
    return a, args, b


def func_multiple_poskw_args_kw(a, b, *args, c, d):
    return a, b, args, c, d


def func_multiple_poskw_args_kw_typed(
    a: a_types[10], b: b_type, *args, c: c_types[2], d: d_type
) -> ((tuple, dict), str, tuple, dict, MyClass):
    return a, b, args, c, d


def func_multiple_poskw_args_kw_defaults(a=a_default, b=b_default, *args, c=c_default, d=d_default):
    return a, b, args, c, d


def func_multiple_poskw_args_kw_typed_defaults(
    a: a_types[11] = a_default, b: b_type = b_default, *args, c: c_types[3] = c_default, d: d_type = d_default
) -> ((tuple, dict), str, tuple, dict, MyClass):
    return a, b, args, c, d


def func_single_poskw_args_kwargs(a, *args, **kwargs):
    return a, args, kwargs


def func_single_poskw_args_kwargs_typed(a: a_types[12], *args, **kwargs) -> ((tuple, dict), tuple, dict):
    return a, args, kwargs


def func_single_poskw_args_kwargs_defaults(a=a_default, *args, **kwargs):
    return a, args, kwargs


def func_single_poskw_args_kwargs_typed_defaults(
    a: a_types[13] = a_default, *args, **kwargs
) -> ((tuple, dict), tuple, dict):
    return a, args, kwargs


def func_multiple_poskw_args_kwargs(a, b, *args, **kwargs):
    return a, b, args, kwargs


def func_multiple_poskw_args_kwargs_typed(
    a: a_types[14], b: b_type, *args, **kwargs
) -> ((tuple, dict), str, dict, MyClass, tuple, dict):
    return a, b, args, kwargs


def func_multiple_poskw_args_kwargs_defaults(a=a_default, b=b_default, *args, **kwargs):
    return a, b, args, kwargs


def func_multiple_poskw_args_kwargs_typed_defaults(
    a: a_types[15] = a_default, b: b_type = b_default, *args, **kwargs
) -> ((tuple, dict), str, dict, MyClass):
    return a, b, args, kwargs


def func_single_poskw_kw_kwargs(a, *, b, **kwargs):
    return a, b, kwargs


def func_single_poskw_kw_kwargs_typed(a: a_types[16], *, b: b_type, **kwargs) -> ((tuple, dict), str, dict):
    return a, b, kwargs


def func_single_poskw_kw_kwargs_defaults(a=a_default, *, b=b_default, **kwargs):
    return a, b, kwargs


def func_single_poskw_kw_kwargs_typed_defaults(
    a: a_types[17] = a_default, *, b: b_type = b_default, **kwargs
) -> ((tuple, dict), str, dict):
    return a, b, kwargs


def func_multiple_poskw_kw_kwargs(a, b, *, c, d, **kwargs):
    return a, b, c, d, kwargs


def func_multiple_poskw_kw_kwargs_typed(
    a: a_types[18], b: b_type, *, c: c_types[0], d: d_type, **kwargs
) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, c, d, kwargs


def func_multiple_poskw_kw_kwargs_defaults(a=a_default, b=b_default, *, c=c_default, d=d_default, **kwargs):
    return a, b, c, d, kwargs


def func_multiple_poskw_kw_kwargs_typed_defaults(
    a: a_types[19] = a_default, b: b_type = b_default, *, c: c_types[1] = c_default, d: d_type = d_default, **kwargs
) -> ((tuple, dict), str, dict, MyClass, dict):
    return a, b, c, d, kwargs


def func_single_args_kw_kwargs(*args, a, **kwargs):
    return args, a, kwargs


def func_single_args_kw_kwargs_typed(*args, a: a_types[20], **kwargs) -> (tuple, (tuple, dict), dict):
    return args, a, kwargs


def func_single_args_kw_kwargs_defaults(*args, a=a_default, **kwargs):
    return args, a, kwargs


def func_single_args_kw_kwargs_typed_defaults(
    *args, a: a_types[21] = a_default, **kwargs
) -> (tuple, (tuple, dict), dict):
    return args, a, kwargs


def func_multiple_args_kw_kwargs(*args, a, b, **kwargs):
    return args, a, b, kwargs


def func_multiple_args_kw_kwargs_typed(
    *args, a: a_types[22], b: b_type, **kwargs
) -> (tuple, (tuple, dict), str, dict, MyClass, dict):
    return args, a, b, kwargs


def func_multiple_args_kw_kwargs_defaults(*args, a=a_default, b=b_default, **kwargs):
    return args, a, b, kwargs


def func_multiple_args_kw_kwargs_typed_defaults(
    *args, a: a_types[23] = a_default, b: b_type = b_default, **kwargs
) -> (tuple, (tuple, dict), str, dict, MyClass, dict):
    return args, a, b, kwargs


def func_single_pos_poskw_args_kw(a, /, b, *args, c):
    return a, b, args, c


def func_single_pos_poskw_args_kw_typed(
    a: a_types[24], /, b: b_type, *args, c: c_types[2]
) -> ((tuple, dict), str, tuple, dict):
    return a, b, args, c


def func_single_pos_poskw_args_kw_defaults(a=a_default, /, b=b_default, *args, c=c_default):
    return a, b, args, c


def func_single_pos_poskw_args_kw_typed_defaults(
    a: a_types[25] = a_default, /, b: b_type = b_default, *args, c: c_types[3] = c_default
) -> ((tuple, dict), str, tuple, dict):
    return a, b, args, c


def func_multiple_pos_poskw_args_kw(a, b, /, c, d, *args, e, f):
    return a, b, c, d, args, e, f


def func_multiple_pos_poskw_args_kw_typed(
    a: a_types[26], b: b_type, /, c: c_types[0], d: d_type, *args, e: e_types[0], f: f_types[2]
) -> ((tuple, dict), str, dict, MyClass, tuple, Union[None, GenericAlias], Any):
    return a, b, c, d, args, e, f


def func_multiple_pos_poskw_args_kw_defaults(
    a=a_default, b=b_default, /, c=c_default, d=d_default, *args, e=e_default, f=f_default
):
    return a, b, c, d, args, e, f


def func_multiple_pos_poskw_args_kw_typed_defaults(
    a: a_types[27] = a_default,
    b: b_type = b_default,
    /,
    c: c_types[1] = c_default,
    d: d_type = d_default,
    *args,
    e: e_types[1] = e_default,
    f: f_types[0] = f_default,
) -> ((tuple, dict), str, dict, MyClass, tuple, Union[None, GenericAlias], Any):
    return a, b, c, d, args, e, f


def func_single_pos_poskw_args_kwargs(a, /, b, *args, **kwargs):
    return a, b, args, kwargs


def func_single_pos_poskw_args_kwargs_typed(
    a: a_types[28], /, b: b_type, *args, **kwargs
) -> ((tuple, dict), str, tuple, dict):
    return a, b, args, kwargs


def func_single_pos_poskw_args_kwargs_defaults(a=a_default, /, b=b_default, *args, **kwargs):
    return a, b, args, kwargs


def func_single_pos_poskw_args_kwargs_typed_defaults(
    a: a_types[29] = a_default, /, b: b_type = b_default, *args, **kwargs
) -> ((tuple, dict), str, tuple, dict):
    return a, b, args, kwargs


def func_multiple_pos_poskw_args_kwargs(a, b, /, c, d, *args, **kwargs):
    return a, b, c, d, args, kwargs


def func_multiple_pos_poskw_args_kwargs_typed(
    a: a_types[30], b: b_type, /, c: c_types[2], d: d_type, *args, **kwargs
) -> ((tuple, dict), str, dict, MyClass, tuple, dict):
    return a, b, c, d, args, kwargs


def func_multiple_pos_poskw_args_kwargs_defaults(
    a=a_default, b=b_default, /, c=c_default, d=d_default, *args, **kwargs
):
    return a, b, c, d, args, kwargs


def func_multiple_pos_poskw_args_kwargs_typed_defaults(
    a: a_types[31] = a_default,
    b: b_type = b_default,
    /,
    c: c_types[3] = c_default,
    d: d_type = d_default,
    *args,
    **kwargs,
) -> ((tuple, dict), str, dict, MyClass, tuple, dict):
    return a, b, c, d, args, kwargs


def func_single_pos_poskw_kw_kwargs(a, /, b, *, c, **kwargs):
    return a, b, c, kwargs


def func_single_pos_poskw_kw_kwargs_typed(
    a: a_types[0], /, b: b_type, *, c: c_types[0], **kwargs
) -> ((tuple, dict), str, dict, dict):
    return a, b, c, kwargs


def func_single_pos_poskw_kw_kwargs_defaults(a=a_default, /, b=b_default, *, c=c_default, **kwargs):
    return a, b, c, kwargs


def func_single_pos_poskw_kw_kwargs_typed_defaults(
    a: a_types[1] = a_default, /, b: b_type = b_default, *, c: c_types[1] = c_default, **kwargs
) -> ((tuple, dict), str, dict, dict):
    return a, b, c, kwargs


def func_multiple_pos_poskw_kw_kwargs(a, b, /, c, d, *, e, f, **kwargs):
    return a, b, c, d, e, f, kwargs


def func_multiple_pos_poskw_kw_kwargs_typed(
    a: a_types[2], b: b_type, /, c: c_types[2], d: d_type, *, e: e_types[0], f: f_types[1], **kwargs
) -> ((tuple, dict), str, dict, MyClass, Union[None, GenericAlias], Any, dict):
    return a, b, c, d, e, f, kwargs


def func_multiple_pos_poskw_kw_kwargs_defaults(
    a=a_default, b=b_default, /, c=c_default, d=d_default, *, e=e_default, f=f_default, **kwargs
):
    return a, b, c, d, e, f, kwargs


def func_multiple_pos_poskw_kw_kwargs_typed_defaults(
    a: a_types[3] = a_default,
    b: b_type = b_default,
    /,
    c: c_types[3] = c_default,
    d: d_type = d_default,
    *,
    e: e_types[1] = e_default,
    f: f_types[2] = f_default,
    **kwargs,
) -> ((tuple, dict), str, dict, MyClass):
    return a, b, c, d, e, f, kwargs


def func_single_pos_args_kw_kwargs(a, /, *args, b, **kwargs):
    return a, args, b, kwargs


def func_single_pos_args_kw_kwargs_typed(
    a: a_types[4], /, *args, b: b_type, **kwargs
) -> ((tuple, dict), tuple, str, dict):
    return a, args, b, kwargs


def func_single_pos_args_kw_kwargs_defaults(a=a_default, /, *args, b=b_default, **kwargs):
    return a, args, b, kwargs


def func_single_pos_args_kw_kwargs_typed_defaults(
    a: a_types[5] = a_default, /, *args, b: b_type, **kwargs
) -> ((tuple, dict), tuple, str, dict):
    return a, args, b, kwargs


def func_multiple_pos_args_kw_kwargs(a, b, /, *args, c, d, **kwargs):
    return a, b, args, c, d, kwargs


def func_multiple_pos_args_kw_kwargs_typed(
    a: a_types[6], b: b_type, /, *args, c: c_types[0], d: d_type, **kwargs
) -> ((tuple, dict), str, tuple, dict, MyClass, dict):
    return a, b, args, c, d, kwargs


def func_multiple_pos_args_kw_kwargs_defaults(a=a_default, b=b_default, /, *args, c=c_default, d=d_default, **kwargs):
    return a, b, args, c, d, kwargs


def func_multiple_pos_args_kw_kwargs_typed_defaults(
    a: a_types[7] = a_default,
    b: b_type = b_default,
    /,
    *args,
    c: c_types[1] = c_default,
    d: d_type = d_default,
    **kwargs,
) -> ((tuple, dict), str, tuple, dict, MyClass, dict):
    return a, b, args, c, d, kwargs


def func_single_poskw_args_kw_kwargs(a, *args, b, **kwargs):
    return a, args, b, kwargs


def func_single_poskw_args_kw_kwargs_typed(
    a: a_types[8], *args, b: b_type, **kwargs
) -> ((tuple, dict), tuple, str, dict):
    return a, args, b, kwargs


def func_single_poskw_args_kw_kwargs_defaults(a=a_default, *args, b=b_default, **kwargs):
    return a, args, b, kwargs


def func_single_poskw_args_kw_kwargs_typed_defaults(
    a: a_types[9] = a_default, *args, b: b_type = b_default, **kwargs
) -> ((tuple, dict), tuple, str, dict):
    return a, args, b, kwargs


def func_multiple_poskw_args_kw_kwargs(a, b, *args, c, d, **kwargs):
    return a, b, args, c, d, kwargs


def func_multiple_poskw_args_kw_kwargs_typed(
    a: a_types[10], b: b_type, *args, c: c_types[2], d: d_type, **kwargs
) -> ((tuple, dict), str, tuple, dict, MyClass, dict):
    return a, b, args, c, d, kwargs


def func_multiple_poskw_args_kw_kwargs_defaults(a=a_default, b=b_default, *args, c=c_default, d=d_default, **kwargs):
    return a, b, args, c, d, kwargs


def func_multiple_poskw_args_kw_kwargs_typed_defaults(
    a: a_types[11] = a_default, b: b_type = b_default, *args, c: c_types[3] = c_default, d: d_type = d_default, **kwargs
) -> ((tuple, dict), str, tuple, dict, MyClass, dict):
    return a, b, args, c, d, kwargs


def func_single_pos_poskw_args_kw_kwargs(a, /, b, *args, c, **kwargs):
    return a, b, args, c, kwargs


def func_single_pos_poskw_args_kw_kwargs_typed(
    a: a_types[12], /, b: b_type, *args, c: c_types[0], **kwargs
) -> ((tuple, dict), str, tuple, dict, dict):
    return a, b, args, c, kwargs


def func_single_pos_poskw_args_kw_kwargs_defaults(a=a_default, /, b=b_default, *args, c=c_default, **kwargs):
    return a, b, args, c, kwargs


def func_single_pos_poskw_args_kw_kwargs_typed_defaults(
    a: a_types[13] = a_default, /, b: b_type = b_default, *args, c: c_types[1] = c_default, **kwargs
) -> ((tuple, dict), str, tuple, dict, dict):
    return a, b, args, c, kwargs


def func_multiple_pos_poskw_args_kw_kwargs(a, b, /, c, d, *args, e, f, **kwargs):
    return a, b, c, d, args, e, f, kwargs


def func_multiple_pos_poskw_args_kw_kwargs_typed(
    a: a_types[14], b: b_type, /, c: c_types[2], d: d_type, *args, e: e_types[0], f: f_types[0], **kwargs
) -> ((tuple, dict), str, dict, MyClass, tuple, Union[None, GenericAlias], Any, dict):
    return a, b, c, d, args, e, f, kwargs


def func_multiple_pos_poskw_args_kw_kwargs_defaults(
    a=a_default, b=b_default, /, c=c_default, d=d_default, *args, e=e_default, f=f_default, **kwargs
):
    return a, b, c, d, args, e, f, kwargs


def func_multiple_pos_poskw_args_kw_kwargs_typed_defaults(
    a: a_types[15] = a_default,
    b: b_type = b_default,
    /,
    c: c_types[3] = c_default,
    d: d_type = d_default,
    *args,
    e: e_types[1] = e_default,
    f: f_types[1] = f_default,
    **kwargs,
) -> ((tuple, dict), str, dict, MyClass, tuple, Union[None, GenericAlias], Any, dict):
    return a, b, c, d, args, e, f, kwargs
