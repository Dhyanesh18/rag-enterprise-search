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

    def log_ingestion(self, file_path, total_chunks, ingested_chunks):
        log_entry = {
            "timestamp": time.strftime("%A, %B %d, %Y at %I:%M %p", time.localtime()),
            "file":file_path,
            "total_chunks":total_chunks,
            "ingested_chunks":ingested_chunks
        }

        log_file = self.file_path

        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)