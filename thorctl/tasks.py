import dataclasses
import json
import uuid
import pika


@dataclasses.dataclass
class TaskPayload:
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
        return cls(**json.loads(s))

    @classmethod
    def from_bytes(cls, b: bytes):
        return cls.from_str(b.decode("utf8"))


class Task:
    def __init__(
            self,
            channel: pika.channel.Channel,
            delivery_tag: int,
            payload: TaskPayload):
        self._channel = channel
        self._delivery_tag = delivery_tag
        self.payload = payload

    @classmethod
    def from_msg(cls,
                 channel: pika.channel.Channel,
                 method: pika.amqp_object.Method,
                 properties: pika.amqp_object.Properties,
                 body: bytes) -> "Task":
        return Task(
            channel=channel,
            delivery_tag=method.delivery_tag,
            payload=TaskPayload.from_bytes(body)
        )

    def mark_success(self):
        self._channel.basic_ack(delivery_tag=self._delivery_tag)

    def mark_failed(self, requeue: bool = False):
        self._channel.basic_nack(
            delivery_tag=self._delivery_tag,
            requeue=requeue,
        )


def new_task_id():
    return str(uuid.uuid4())
