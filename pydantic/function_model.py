"""Provides FunctionModel, a model for validating function arguments."""

from copy import deepcopy
from inspect import Parameter, Signature, signature
from types import NoneType
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar, Union

from pydantic_core import InitErrorDetails, PydanticCustomError, PydanticKnownError, PydanticUndefined, ValidationError

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model, errors


class FunctionModel:
    """A class for validating function arguments and calling the function upon the validated arguments."""

    function: Callable
    _model: Type[BaseModel]
    _parsed: Optional[BaseModel]
    _return: TypeAdapter
    _parameters: dict
    _keyword_total: int
    _positional_total: int

    T = TypeVar('T')

    def __init__(self, function: Callable) -> None:
        """Create a new model based on the provided function.

        Args:
            function: The function that the model should encapsulate.
        """
        self.function = function
        self._parameters = {}
        self._parsed = None
        self._keyword_total = 0
        self._positional_total = 0

        # Gather all parameters from the function
        params = {}  # In order, as params[<name>] = (type, default)
        has_args = False
        has_kwargs = False
        sig = signature(function)
        for name, param in sig.parameters.items():
            if name == 'args' and param.kind != Parameter.VAR_POSITIONAL:
                raise ValueError('The name "args" can only be used for variables in the case of *args.')
            elif name == 'kwargs' and param.kind != Parameter.VAR_KEYWORD:
                raise ValueError('The name "kwargs" can only be used for variables in the case of **kwargs.')
            elif name.startswith('_'):
                raise _internal_key_error(name)

            if param.kind == Parameter.VAR_POSITIONAL:
                has_args = True
            elif param.kind == Parameter.VAR_KEYWORD:
                has_kwargs = True
            else:
                is_positional = param.kind == Parameter.POSITIONAL_ONLY or param.kind == Parameter.POSITIONAL_OR_KEYWORD
                is_keyword = param.kind == Parameter.KEYWORD_ONLY or param.kind == Parameter.POSITIONAL_OR_KEYWORD

                if is_positional:
                    self._positional_total += 1
                if is_keyword:
                    self._keyword_total += 1

                params[name] = (
                    get_annotation_type(param.annotation),
                    param.default if param.default != Parameter.empty else PydanticUndefined,
                )
                self._parameters[name] = {
                    'positional': is_positional,
                    'keyword': is_keyword,
                    'set_keyword': None,  # True for set using keyword, False for set using position
                }

        # Configure return value type checking
        ret = get_annotation_type(sig.return_annotation)
        self._return = TypeAdapter(ret)

        # Allow extra params if there is a **kwargs argument
        config = (
            ConfigDict(extra='allow', validate_assignment=True, validate_default=True)
            if has_kwargs
            else ConfigDict(extra='forbid', validate_assignment=True, validate_default=True)
        )

        # Handle *args
        if has_args:
            params['args'] = (list, None)

        self._model = create_model(f'{function.__name__}Model', **params, __config__=config)

    def parse_arguments(self, *args: Any, **kwargs: Any):
        """Parse the provided arguments into the model's parameters.

        Args:
            *args: Any positional arguments to parse (will fit positional-only and positional-or-keyword parameters).
            **kwargs: Any keyword arguments to parse (will fit keyword-only and positional-or-keyword parameters).
        """
        parsed, metadata = self.validate_arguments(*args, **kwargs)

        self._parsed = parsed
        self._parameters = metadata

    def update_argument(self, pos_or_kw: Union[int, str], value: Any) -> None:
        """Update an initialized argument.

        Args:
            pos_or_kw: A position or keyword that identifies the intended parameter.
            value: The value to set the argument to.
        """
        if self._parsed is None:
            raise AttributeError('Arguments have not been initialized, so they cannot be updated.')

        try:
            key, details = self._get_parameter(pos_or_kw)
        except (PydanticKnownError, PydanticCustomError) as error:
            if isinstance(error, PydanticKnownError):
                error = error.type

            init: InitErrorDetails = {
                'type': error,
                'loc': (pos_or_kw,),
                'input': value,
            }
            raise ValidationError.from_exception_data(self.__class__.__name__, [init])

        if key == 'args':
            assert isinstance(pos_or_kw, int)  # Should always be true, otherwise something is broken

            args_pos = pos_or_kw - self._positional_total
            args = getattr(self._parsed, 'args')
            args_len = len(args)
            if args_len <= args_pos - 1:
                # The previous positional argument hasn't been set yet, so we can't set this one
                raise AttributeError(
                    f'Cannot set argument at position {pos_or_kw}, as the preceding argument has not been set.'
                )
            elif args_len <= args_pos:
                # This argument hasn't been set yet, but the preceding one has, so append it
                args.append(value)
            else:
                args[args_pos] = value

        else:
            setattr(self._parsed, key, value)

    def validate_argument(self, pos_or_kw: Union[int, str], value: Any) -> Any:
        """Validate an argument against a specific parameter.

        Args:
            pos_or_kw: A position or keyword that identifies the intended parameter.
            value: The value to validate as a valid parameter input.
        """
        try:
            key, details = self._get_parameter(pos_or_kw)
        except (PydanticCustomError, PydanticKnownError) as error:
            if isinstance(error, PydanticKnownError):
                error = error.type

            init: InitErrorDetails = {
                'type': error,
                'loc': (pos_or_kw,),
                'input': value,
            }
            raise ValidationError.from_exception_data(self.__class__.__name__, [init])

        if key == 'args':
            # Values within *args are unrestricted
            return value
        elif key == 'kwargs':
            # Values within **kwargs are unrestricted
            return value
        else:
            field = self._model.model_fields.get(key)
            assert field is not None  # This should always be the case, but Python complains otherwise
            return TypeAdapter(get_annotation_type(field.annotation)).validate_python(value)

    def validate_arguments(self, *args: Any, **kwargs: Any) -> Tuple[BaseModel, dict]:  # noqa: C901
        """Validate a set of arguments against the model's parameters.

        Raises errors in response to invalid arguments.

        Args:
            args: The positional arguments to validate.
            kwargs: The keyword arguments to validate.

        Returns:
            The validated arguments, separated into general arguments and *args.
        """
        param_keys = list(self._parameters.keys())
        params = deepcopy(self._parameters)
        errors: List[InitErrorDetails] = []
        validated = {}

        # Reset set_keyword metadata for this run
        for key in param_keys:
            params[key]['set_keyword'] = None

        # Validate keyword arguments match keyword parameters
        has_kwargs = self._model.model_config.get('extra') == 'allow'
        for kw, value in kwargs.items():
            unexp_error: InitErrorDetails = {
                'type': 'unexpected_keyword_argument',
                'loc': (kw,),
                'input': value,
            }
            error: Optional[InitErrorDetails] = None

            if kw == 'args' or kw == 'kwargs' or kw.startswith('_'):
                # Keyword for internal use only
                error = {
                    'type': _internal_key_error(kw),
                    'loc': (kw,),
                    'input': value,
                }
            elif kw not in param_keys:
                if not has_kwargs:
                    error = unexp_error  # Unknown keyword and **kwargs isn't present
            elif not self._parameters[kw]['keyword']:
                error = unexp_error  # This parameter is positional-only, so can't be specified by keyword

            if error:
                errors.append(error)
            else:
                # No error, so add this to the validated arguments
                validated[kw] = value
                if kw in param_keys:
                    params[kw]['set_keyword'] = True

        # Validate positional arguments
        has_args = 'args' in self._model.model_fields.keys()
        val_keys = validated.keys()
        extra_args = []
        for pos in range(len(args)):
            value = args[pos]
            unexp_error: InitErrorDetails = {
                'type': 'unexpected_positional_argument',
                'loc': (pos,),
                'input': value,
            }
            error: Optional[InitErrorDetails] = None
            in_args = False

            if pos >= len(param_keys):
                if has_args:
                    in_args = True
                else:
                    error = unexp_error  # Unknown positional parameter and *args is not present
            elif not self._parameters[param_keys[pos]]['positional']:
                if has_args:
                    in_args = True
                else:
                    error = unexp_error  # Keyword-only parameter, so can't set via position
            elif param_keys[pos] in val_keys:
                error = {
                    'type': 'multiple_argument_values',
                    'loc': (pos, param_keys[pos]),
                    'input': value,
                }

            if error:
                errors.append(error)
            elif in_args:
                extra_args.append(value)
            else:
                kw = param_keys[pos]
                validated[kw] = value
                params[kw]['set_keyword'] = False

        if has_args:
            # Include *args values in model
            validated['args'] = extra_args

        parsed = self._model(**validated)

        # If any arguments were set by their defaults, fix their `set_keyword` metadata
        for key, value in parsed.model_fields.items():
            if key in param_keys and params[key]['set_keyword'] is None:
                param = params[key]
                param['set_keyword'] = True if param['keyword'] else False

        if len(errors) > 0:
            raise ValidationError.from_exception_data(self.__class__.__name__, errors)
        else:
            return parsed, params

    def get_argument(self, pos_or_kw: Union[int, str]) -> Any:
        """Retrieve the current argument provided for a given parameter.

        Args:
            pos_or_kw: A position or keyword that identifies the intended parameter.

        Returns:
            The current value of the specified argument.
        """
        if self._parsed is None:
            raise AttributeError('Arguments have not been set, so arguments cannot be retrieved.')

        key, details = self._get_parameter(pos_or_kw)

        if key == 'args':
            assert isinstance(pos_or_kw, int)  # This should always be the case, otherwise something is broken

            # Get argument from within *args
            args_pos = pos_or_kw - self._positional_total
            args = getattr(self._parsed, 'args')
            if len(args) > args_pos:
                return args[args_pos]
            else:
                raise AttributeError(f'Attempted to access non-existent positional argument (position {pos_or_kw}).')
        elif key == 'kwargs':
            assert isinstance(pos_or_kw, str)  # This should always be the case, otherwise something is broken
            return getattr(self._parsed, pos_or_kw)
        else:
            return getattr(self._parsed, key)

    def get_arguments(self) -> Tuple[list, dict]:
        """Retrieve the current arguments in the model.

        Returns:
            The current arguments stored in the model, in the form (positional, keyword).
        """
        if self._parsed is None:
            # Can't run function on non-existent arguments
            raise AttributeError('No arguments have yet been parsed, so the function cannot be called upon them.')

        positional = []
        args = []
        keyword = {}

        param_keys = self._parameters.keys()
        for key, value in self._parsed.model_dump().items():
            if key == 'args':
                args = value
            elif key not in param_keys or self._parameters[key]['set_keyword']:
                keyword[key] = value
            else:
                positional.append(value)

        return positional + args, keyword

    def call(self, function: Optional[Callable[..., T]] = None) -> T:
        """Calls the model's function with its stored arguments.
        Can optionally be used to call a different, provided function with these arguments.

        Args:
            function: The function to be called using the model's arguments.
                By default, the model's initially provided function will be used.

        Returns:
            The function's return value, if any.
        """
        if function is None:
            function = self.function

        positional, keyword = self.get_arguments()

        return self._return.validate_python(function(*positional, **keyword))

    def _get_parameter(self, pos_or_kw: Union[int, str]) -> tuple[str, dict]:
        """Retrieves the parameter details for the indexed parameter.

        Args:
            pos_or_kw: A position or keyword that identifies the intended parameter.

        Returns:
            The name and details of the selected parameter.
        """
        if isinstance(pos_or_kw, int):
            index_error = PydanticKnownError('unexpected_positional_argument')
            param_keys = list(self._parameters.keys())

            # Check if the index is outside of the range of positional parameters
            if pos_or_kw >= self._positional_total or not self._parameters[param_keys[pos_or_kw]]['positional']:
                if 'args' in self._model.model_fields.keys():
                    return 'args', {'positional': True, 'keyword': False}
                else:
                    raise index_error

            key = param_keys[pos_or_kw]
            return key, self._parameters[key]

        else:
            # Handle *args and **kwargs calls - invalid as not possible on a function, and possibly breaking internally
            if pos_or_kw == 'args' or pos_or_kw == 'kwargs' or pos_or_kw.startswith('_'):
                raise _internal_key_error(pos_or_kw)

            # Check if the key would be within **kwargs and whether that would be valid
            key_error = PydanticKnownError('unexpected_keyword_argument')
            try:
                param = self._parameters[pos_or_kw]
            except KeyError:
                if self._model.model_config.get('extra') == 'allow':
                    return 'kwargs', {'positional': False, 'keyword': True}
                else:
                    raise key_error

            if not param['keyword']:
                raise key_error

            return pos_or_kw, param


def get_annotation_type(annotation: Any, strict: bool = False) -> Type:
    """Transforms a given annotation into a Type that is compatible with Pydantic, if possible.

    Args:
        annotation: The original annotation.
        strict: Whether to raise an exception if the annotation is incompatible, rather than returning `Any`.

    Returns:
        The transformed, Pydantic-compatible Type. `Any` if completely incompatible.
    """
    if annotation == Parameter.empty or annotation == Signature.empty:
        # These are provided if an annotation is undefined in a function's signature
        return Any
    elif annotation is None:
        # You can annotate a function with this and it will work, but passing None to a model doesn't work
        return NoneType
    elif isinstance(annotation, tuple):
        # Convert `(a, b)` annotations into `tuple[a, b]`, as the former is incompatible. Done recursively.
        anno = ()
        for ty in annotation:
            if ty is ...:
                anno = anno + (...,)
            else:
                anno = anno + (get_annotation_type(ty),)
        return Tuple[anno] if anno != () else Tuple  # type: ignore
    else:
        # Return the annotation as-is if it is compatible, otherwise return Any
        try:
            TypeAdapter(annotation)
            return annotation
        except errors.PydanticSchemaGenerationError:
            if strict:
                raise PydanticCustomError('unknown_type', f'the type "{annotation}" is incompatible with Pydantic')  # type: ignore
            else:
                return Any


def _internal_key_error(key: str):
    return PydanticCustomError('internal_key', f'the key "{key}" is reserved for internal use only')  # type: ignore
