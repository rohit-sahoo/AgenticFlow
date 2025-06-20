from app.tools.code_executor import CodeExecutor

def test_code_executor_hello_world():
    executor = CodeExecutor()
    code = "print('Hello World')"
    result = executor.run(code)
    assert "Hello World" in result

def test_code_executor_variable():
    executor = CodeExecutor()
    code = "a = 5\nb = 10\nprint(a + b)"
    result = executor.run(code)
    assert "15" in result 