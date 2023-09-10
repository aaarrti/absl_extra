from absl_extra.flax_utils import TrainingHooks, combine_hooks


def func1(*args, **kwargs):
    print("1")


def func2(*args, **kwargs):
    print("2")


def func3(*args, **kwargs):
    print("3")


def test_combine_hooks():
    hooks1 = TrainingHooks(on_step_end=[func1, func2])

    hooks2 = TrainingHooks(on_step_end=[func3])

    hooks = combine_hooks(hooks1, hooks2)

    assert hooks.on_step_end == [func1, func2, func3]
