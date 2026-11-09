"""Validation of requested length values; this does not analyze generated text."""

import math


def number(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number.')
    if integer and not isinstance(value, int):
        raise ValueError(f'{label} must be an integer.')
    return value


def bounds(value, label, minimum=None, maximum=None, integer=False, open_max=False):
    if not isinstance(value, dict) or set(value) != {'min', 'max'}:
        raise ValueError(f'{label} requires both min and max.')
    low = number(value['min'], label, integer)
    high = value['max']
    if high is None and open_max:
        if minimum is not None and low < minimum:
            raise ValueError(f'{label} must be at least {minimum}.')
        if maximum is not None:
            raise ValueError(f'{label} needs a maximum no greater than {maximum}.')
        return
    high = number(high, label, integer)
    if low > high or (minimum is not None and low < minimum) or (maximum is not None and high > maximum):
        raise ValueError(f'{label} is outside the permitted bounds or has min greater than max.')


def validate_length(values, specification):
    allowed = specification['controls']
    if not isinstance(values, dict) or set(values) - set(allowed):
        raise ValueError('The system template does not expose these length controls.')
    resolved = {}
    for key, settings in allowed.items():
        value = values.get(key, settings.get('default'))
        if value is None:
            if not settings.get('allow_disable', False):
                raise ValueError(f'{key} cannot be disabled.')
        else:
            limits = settings.get('value_bounds', {})
            target = {'min': value, 'max': value} if key == 'max_word_length' else value
            bounds(target, key, limits.get('minimum'), limits.get('maximum'), integer=True)
        resolved[key] = value
    return resolved
