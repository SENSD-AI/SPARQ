from sparq.tools.python_repl.namespace import clean_namespace


def test_clean_namespace_preserves_modules():
    namespace = {
        "__builtins__": object(),
        "__name__": "x",
        "__modules__": {"np": "numpy"},
        "user_var": 123,
    }

    clean_namespace(namespace)

    assert "__builtins__" not in namespace
    assert "__name__" not in namespace
    assert "__modules__" in namespace
    assert namespace["__modules__"] == {"np": "numpy"}
    assert namespace["user_var"] == 123