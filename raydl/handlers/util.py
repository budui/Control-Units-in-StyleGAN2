import functools

from loguru import logger


def disable_if_failed(func):
    """
    useful Decorator for ignite handlers. if some handler run failed, do not run this handler in next trigger event.
    :param func: handler
    :return: handler
    """
    _enable = True

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal _enable
        if _enable:
            try:
                func(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                # Catch all Exception, log error and disable handler.
                _enable = False
                logger.warning(f"disable {func.__name__} due to error below")
                logger.exception(e)
        else:
            logger.debug(f"skip {func.__name__} because it failed in the last execution")

    return wrapper
