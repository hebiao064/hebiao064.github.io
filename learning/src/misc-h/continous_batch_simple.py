import torch
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import time
import random
import threading


class ForwardMode(Enum):
    PREFILL = "PREFILL"
    DECODE = "DECODE"


@dataclass
class Req:
    id: int
    input_ids: List[int]
    output_ids: List[int]
    max_new_tokens: int
    is_finished: bool = False


@dataclass
class ScheduleBatch:
    reqs: List[Req]
    forward_mode: ForwardMode


class Scheduler:
    def __init__(self):
        self.waiting_queue = []
        self.running_batch = []
        self.last_batch: Optional[ScheduleBatch] = None

    def add_request(self, req: Req):
        self.waiting_queue.append(req)

    def start(self):
        def event_loop():
            while True:
                batch = self.get_next_batch()

                if batch:
                    output = self.run_batch(batch)
                    self.process_batch_output(batch, output)
                else:
                    time.sleep(0.1)
                    # print("No batch available, sleeping for 0.1 seconds")

                self.last_batch = batch

        thread = threading.Thread(target=event_loop)
        thread.daemon = True
        thread.start()

    def get_next_batch(self):
        # update last prefill batch to be decode batch
        if self.last_batch and self.last_batch.forward_mode == ForwardMode.PREFILL:
            self.running_batch.extend(self.last_batch.reqs)

        # if prefill exist, then make a prefill batch
        if self.waiting_queue:
            new_prefill_batch = ScheduleBatch(
                reqs=self.waiting_queue, forward_mode=ForwardMode.PREFILL
            )
            self.waiting_queue = []
            return new_prefill_batch

        if self.running_batch:
            new_decode_batch = ScheduleBatch(
                reqs=self.running_batch, forward_mode=ForwardMode.DECODE
            )
            return new_decode_batch

    def run_batch(self, batch: ScheduleBatch):
        print(
            f"Running batch with {len(batch.reqs)} requests in {batch.forward_mode} mode"
        )
        time.sleep(0.4)
        return torch.randint(0, 100, (len(batch.reqs),)).tolist()

    def process_batch_output(self, batch: ScheduleBatch, next_output_ids: List[int]):
        for req, next_output_id in zip(batch.reqs, next_output_ids):
            req.output_ids.append(next_output_id)
            if len(req.output_ids) >= req.max_new_tokens:
                print(
                    f"Request {req.id} finished with input_ids: {req.input_ids} and output_ids: {req.output_ids}"
                )
                req.is_finished = True

        self.running_batch = [req for req in self.running_batch if not req.is_finished]


scheduler = Scheduler()
scheduler.start()


for i in range(10):
    time.sleep(random.uniform(0.1, 0.3))
    scheduler.add_request(
        Req(
            id=i,
            input_ids=[random.randint(0, 100) for _ in range(5)],
            output_ids=[],
            max_new_tokens=random.randint(5, 10),
        )
    )
time.sleep(5)
