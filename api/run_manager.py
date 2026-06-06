import threading
import queue
import uuid
import sys
import io

class RunManager:
    def __init__(self):
        self.active_runs = {} # {run_id: {"thread": th, "stop_event": evt, "queue": q}}
        self.lock = threading.Lock()

    def create_run(self):
        run_id = str(uuid.uuid4())
        with self.lock:
            self.active_runs[run_id] = {
                "stop_event": threading.Event(),
                "queue": queue.Queue(),
                "status": "initializing"
            }
        return run_id

    def get_queue(self, run_id):
        with self.lock:
            return self.active_runs.get(run_id, {}).get("queue")

    def stop_run(self, run_id):
        with self.lock:
            if run_id in self.active_runs:
                self.active_runs[run_id]["stop_event"].set()
                return True
        return False

    def cleanup_run(self, run_id):
        with self.lock:
            if run_id in self.active_runs:
                del self.active_runs[run_id]

class OutputInterceptor(io.StringIO):
    """Intercepts stdout and pushes lines to a queue."""
    def __init__(self, q):
        super().__init__()
        self.q = q

    def write(self, s):
        if s.strip():
            self.q.put(s.strip())
        # Still write to actual stdout for terminal debugging, UNLESS it's the massive JSON blob
        if "FINAL_JSON:" not in s:
            sys.__stdout__.write(s)
            sys.__stdout__.flush()

def run_wrapper(run_id, manager, target_func, *args, **kwargs):
    """Wraps the GA execution to capture output and handle events."""
    run_info = manager.active_runs.get(run_id)
    if not run_info: return

    q = run_info["queue"]
    stop_event = run_info["stop_event"]
    
    # Intercept stdout
    original_stdout = sys.stdout
    sys.stdout = OutputInterceptor(q)
    
    try:
        # Pass the stop_event if the target function supports it
        # Otherwise, the GA loop needs to check sys.stdout or a global flag
        kwargs['stop_event'] = stop_event 
        target_func(*args, **kwargs)
    except Exception as e:
        q.put(f"ERROR: {str(e)}")
    finally:
        sys.stdout = original_stdout
        q.put("EOF") # Signals end of stream

run_manager = RunManager()
