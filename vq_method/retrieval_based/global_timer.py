import time
import torch


class Timer:
    def __init__(self, layer_cnt = 32):
        self.pq_compute_time = 0

        self.transfer_time = 0

        self.compute_time = 0

        self.layer_cnt = layer_cnt

        self.decode_pq_start = []
        self.decode_pq_end = []

        self.can_recording = False

        self.transfer_time_tuples = []

    def append_compute_event(self, event_s, event_e):
        self.decode_pq_start.append(event_s)
        assert len(self.decode_pq_start) <= self.layer_cnt
        
        self.decode_pq_end.append(event_e)
        assert len(self.decode_pq_end) <= self.layer_cnt
    
    def set_start_end_event(self, s, e):
        self.start_event = s
        self.end_event = e

    def get_decode_time_parts(self,):
        pq = 0
        non_pq = 0
        torch.cuda.synchronize()
        for i in range(self.layer_cnt):
            pq += self.decode_pq_start[i].elapsed_time(self.decode_pq_end[i])

        for i in range(1, self.layer_cnt):
            non_pq += self.decode_pq_end[i-1].elapsed_time(self.decode_pq_start[i])
        
        non_pq += self.start_event.elapsed_time(self.decode_pq_start[0])
        non_pq += self.decode_pq_end[self.layer_cnt-1].elapsed_time(self.end_event)
        
        transfer_time = 0

        for t in self.transfer_time_tuples:
            time = t[0].elapsed_time(t[1])
            transfer_time += time

        self.transfer_time_tuples = []

        return pq, non_pq, transfer_time, self.start_event.elapsed_time(self.end_event)

    def append_transfer_time_tuples(self, a, b):
        self.transfer_time_tuples.append((a,b))

    def set_recording_state(self, can_recording):
        self.can_recording = can_recording

    def can_record(self):
        return self.can_recording

global_timer = Timer()