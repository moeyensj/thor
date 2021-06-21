import dataclasses
import json
import uuid


@dataclasses.dataclass
class Task:
    job_id: str
    task_id: str
    input_data_uri: str
    result_uri: str

    thorctl_version: str = "0.1.0"

    def to_str(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def to_bytes(self) -> bytes:
        return self.to_str().encode("utf8")

    @classmethod
    def from_str(cls, s: str):
        return Task(**json.loads(s))

    @classmethod
    def from_bytes(cls, b: bytes):
        return cls.from_str(b.decode("utf8"))


def new_task_id():
    return str(uuid.uuid4())
