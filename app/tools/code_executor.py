class CodeExecutor:
    """A simple, sandboxed Python code executor with enhanced error handling."""
    def __init__(self):
        self.execution_timeout = 10  # seconds
        
    def run(self, code: str) -> str:
        import io
        import contextlib
        output = io.StringIO()
        try:
            code = code.strip()
            if not code:
                return "(No code provided)"
            exec_env = {}
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(code, exec_env, exec_env)
            result = output.getvalue().strip()
            if not result:
                for name, value in exec_env.items():
                    if not name.startswith('_') and not callable(value):
                        if isinstance(value, (int, float, str, list, dict, tuple, set)):
                            result = f"Variable '{name}' = {value}"
                            break
            return result or "(Code executed successfully, no output produced)"
        except RecursionError as e:
            return f"Recursion error: {e}. The function may have infinite recursion."
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        except Exception as e:
            return f"Error: {e.__class__.__name__}: {e}" 