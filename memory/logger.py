import json
import os
import time

class JSONLogger:
    def __init__(self, file_path="chat_log.json"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)  # Initialize empty list

    def log(self, user_msg, assistant_msg):
        entry = {
            "user": user_msg,
            "assistant": assistant_msg
        }
        with open(self.file_path, "r+") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
