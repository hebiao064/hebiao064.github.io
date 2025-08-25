import torch
from dataclasses import dataclass, field
from typing import List, Optional
import threading
from enum import Enum
import time
import random

class ForwardMode(Enum):
    PREFILL = "PREFILL"
    DECODE = "DECODE"

@dataclass
class Req:
    rid: int
    origin_input_ids: List[int]
    max_new_tokens: int
    output_ids: List[int] = field(default_factory=list) # optional when init
    is_finished: bool = False

    def length(self) -> int:
        return len(self.origin_input_ids) + len(self.output_ids)

@dataclass
class ScheduleBatch:
    reqs: List[Req]
    forward_mode: ForwardMode


@dataclass
class GenerationBatchResult:
    next_token_ids: Optional[List[int]]


class Scheduler:
    def __init__(self):
        self.waiting_queue = []
        self.running_batch = [] # Decode Batch
        self.last_batch: Optional[ScheduleBatch] = None # The last forward batch

    # Mock the process of tokenizer manager send request to scheduler
    def add_request(self, request):
        self.waiting_queue.append(request)

    def get_next_batch(self) -> Optional[ScheduleBatch]:
        # 1. Merge last Prefill batch into running batch if it exists
        if self.last_batch and self.last_batch.forward_mode == ForwardMode.PREFILL:
            self.running_batch.extend(self.last_batch.reqs) # merge to decode batch

        # 2. Get new Prefill batch and run it if there is any
        if self.waiting_queue:
            # In actual SGLang, we check whether the remaining tokens are enough for this batch
            new_reqs = self.waiting_queue[:]
            self.waiting_queue = []
            new_batch = ScheduleBatch(reqs=new_reqs, forward_mode=ForwardMode.PREFILL)
            return new_batch

        # 3. Run Decode batch
        if self.running_batch:
            # In actual SGLang code, we do retract here if the remaining tokens are not enough for this batch
            return ScheduleBatch(reqs=self.running_batch, forward_mode=ForwardMode.DECODE)

        return None

    def run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
        print(f"Running batch with forward mode: {batch.forward_mode} and batch size: {len(batch.reqs)}")
        print(f"Batch requests ids: {[req.rid for req in batch.reqs]}")

        # Mock the GPU Forward time
        time.sleep(0.4)

        return GenerationBatchResult(next_token_ids=torch.randint(0, 100, (len(batch.reqs),)))

    def process_batch_result(self, batch: ScheduleBatch, result: GenerationBatchResult):
        for req, next_token_id in zip(batch.reqs, result.next_token_ids):
            req.output_ids.append(next_token_id)
            print(f"Request {req.rid} updated with origin input ids: {req.origin_input_ids} and output ids: {req.output_ids}")

            if len(req.output_ids) >= req.max_new_tokens:
                req.is_finished = True
                print(f"Request {req.rid} finished")

        self.running_batch = [req for req in self.running_batch if not req.is_finished]

    def start(self):
        def event_loop():
            while True:
                # 1. Get next batch
                batch = self.get_next_batch()

                # 2. Run Batch
                if batch:
                    result = self.run_batch(batch) 
                    self.process_batch_result(batch, result)
                else:
                    print("No batch to run, sleep 0.1s")
                    time.sleep(0.1) # sleep on IDLE


                # 3. Update last batch which is needed for next get_next_batch()
                self.last_batch = batch

        thread = threading.Thread(target=event_loop)
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    scheduler = Scheduler()
    scheduler.start()

    # we send request randomly to mimic the online traffic
    for i in range(10):
        time.sleep(random.uniform(0.1, 0.3))
        scheduler.add_request(
            Req(rid=i, 
                origin_input_ids=torch.randint(0, 100, (random.randint(1, 5),)).tolist(),
                max_new_tokens=random.randint(1, 5))
        )

    time.sleep(5)