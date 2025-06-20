import os
from datetime import datetime

class FlowLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"log_flow_{timestamp}.txt")
        with open(self.log_path, "w") as f:
            f.write(f"Flow Log started at {timestamp}\n\n")

    def log(self, message: str):
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def log_event(self, event: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"{timestamp} - [EVENT] - {event}")

    def log_user_input(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"{timestamp} - [USER INPUT] - Input from user")

    def log_tool(self, tool_name: str, message: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{timestamp} - [TOOL: {tool_name}]"
        if message:
            msg += f" - {message}"
        self.log(msg)

    def log_agent(self, agent_name: str, message: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{timestamp} - [AGENT: {agent_name}]"
        if message:
            msg += f" - {message}"
        self.log(msg)

    def log_final_response(self, message: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{timestamp} - [FINAL AGENT RESPONSE]"
        if message:
            msg += f" - {message}"
        self.log(msg)

    def log_step_start(self, step_num, agent_name):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"{timestamp} - [EVENT] - Step {step_num} started: {agent_name}")

    def log_step_end(self, step_num, agent_name):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"{timestamp} - [EVENT] - Step {step_num} finished: {agent_name}")

    def log_step(self, step_num, agent_name, input_val, output_val, status):
        self.log(f"[Step {step_num}] Agent: {agent_name}\nInput: {input_val}\nOutput: {output_val}\nStatus: {status}\n---") 