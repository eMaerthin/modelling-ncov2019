import json
from pathlib import Path

class PathlibFriendlyEncoder(json.JSONEncoder):
    """this helps with paths"""
    def default(self, z):
        if isinstance(z, Path):
            return str(z.expanduser())
        else:
            return super().default(z)

class AllString(json.JSONEncoder):
    def default(self, z):
        return str(z)
