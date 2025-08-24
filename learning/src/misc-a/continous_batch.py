import time
import threading
from typing import List, Optional
from dataclasses import dataclass
import torch

@dataclass
class Request:
    rid: str
    input_ids: List[int]
    max_tokens: int
    generated_tokens: List[int] = None
    
    def __post_init__(self):
        if self.generated_tokens is None:
            self.generated_tokens = []
    
    def is_finished(self) -> bool:
        return len(self.generated_tokens) >= self.max_tokens or (
            self.generated_tokens and self.generated_tokens[-1] == 2)  # EOS
    
    def total_length(self) -> int:
        return len(self.input_ids) + len(self.generated_tokens)

class DynamicBatchScheduler:
    def __init__(self, max_batch_size=16, max_total_tokens=4096):
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.waiting_queue: List[Request] = []
        self.running_batch: List[Request] = []
        self.running = False
    
    def add_request(self, rid: str, input_ids: List[int], max_tokens=50):
        """Add new request to waiting queue"""
        req = Request(rid, input_ids, max_tokens)
        self.waiting_queue.append(req)
        print(f"Added request {rid}")
    
    def _can_fit_in_batch(self, new_reqs: List[Request]) -> bool:
        """Check if requests fit memory and batch size constraints"""
        total_reqs = len(self.running_batch) + len(new_reqs)
        if total_reqs > self.max_batch_size:
            return False
        
        total_tokens = sum(req.total_length() + req.max_tokens 
                          for req in self.running_batch + new_reqs)
        return total_tokens <= self.max_total_tokens
    
    def _select_new_requests(self) -> List[Request]:
        """Select requests from waiting queue that can fit in current batch"""
        selected = []
        for req in self.waiting_queue[:]:
            if self._can_fit_in_batch(selected + [req]):
                selected.append(req)
                self.waiting_queue.remove(req)
            else:
                break
        return selected
    
    def _create_batch_tensor(self, requests: List[Request]) -> torch.Tensor:
        """Create padded tensor for batch inference"""
        if not requests:
            return torch.empty(0)
        
        max_len = max(req.total_length() for req in requests)
        batch = torch.zeros(len(requests), max_len, dtype=torch.long)
        
        for i, req in enumerate(requests):
            tokens = req.input_ids + req.generated_tokens
            batch[i, :len(tokens)] = torch.tensor(tokens)
        
        return batch
    
    def _run_inference(self, batch_tensor: torch.Tensor) -> List[int]:
        """Simulate model inference - returns next token for each request"""
        time.sleep(0.01)  # Simulate inference time
        batch_size = batch_tensor.shape[0]
        
        # Generate dummy next tokens (in reality: model(batch_tensor))
        next_tokens = []
        for i in range(batch_size):
            req = self.running_batch[i]
            if len(req.generated_tokens) >= req.max_tokens - 1:
                next_tokens.append(2)  # EOS
            else:
                next_tokens.append(torch.randint(10, 1000, (1,)).item())
        
        return next_tokens
    
    def _update_and_filter_requests(self, next_tokens: List[int]) -> List[Request]:
        """Update requests with new tokens and return finished ones"""
        finished = []
        still_running = []
        
        for req, token in zip(self.running_batch, next_tokens):
            req.generated_tokens.append(token)
            
            if req.is_finished():
                finished.append(req)
                print(f"Request {req.rid} finished with {len(req.generated_tokens)} tokens")
            else:
                still_running.append(req)
        
        self.running_batch = still_running
        return finished
    
    def _step(self):
        """Single scheduler step - core of dynamic batching"""
        # 1. Add new requests to running batch
        new_requests = self._select_new_requests()
        self.running_batch.extend(new_requests)
        
        if not self.running_batch:
            return
        
        print(f"Running batch size: {len(self.running_batch)}, "
              f"Waiting: {len(self.waiting_queue)}")
        
        # 2. Create batch tensor
        batch_tensor = self._create_batch_tensor(self.running_batch)
        
        # 3. Run inference
        next_tokens = self._run_inference(batch_tensor)
        
        # 4. Update requests and remove finished ones
        finished_requests = self._update_and_filter_requests(next_tokens)
    
    def start(self):
        """Start scheduler loop"""
        self.running = True
        
        def scheduler_loop():
            while self.running:
                try:
                    self._step()
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Scheduler error: {e}")
        
        thread = threading.Thread(target=scheduler_loop)
        thread.daemon = True
        thread.start()
        print("Scheduler started")
    
    def stop(self):
        self.running = False
        print("Scheduler stopped")
    
    def get_stats(self):
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_batch),
            "total_tokens": sum(req.total_length() for req in self.running_batch)
        }

# Demo usage
if __name__ == "__main__":
    scheduler = DynamicBatchScheduler(max_batch_size=8, max_total_tokens=1024)
    scheduler.start()
    
    # Add requests over time
    for i in range(6):
        input_ids = [1, 100+i, 200+i, 300+i]
        scheduler.add_request(f"req_{i}", input_ids, max_tokens=10)
        time.sleep(0.2)
    
    # Let it process
    time.sleep(3)
    print(f"Final stats: {scheduler.get_stats()}")
    scheduler.stop()