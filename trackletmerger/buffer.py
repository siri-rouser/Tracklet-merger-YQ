from typing import List

from visionapi_yq.messages_pb2 import SaeMessage


class MessageBuffer:
    '''Keeps messages always sorted by timestamp.'''

    def __init__(self) -> None:
        self._messages: List[SaeMessage] = []

    def append(self, msg: SaeMessage) -> None:
        self._messages.append(msg)
        self._messages.sort(key=lambda m: m.frame.timestamp_utc_ms) # sort the message based on the timestamp
    
    def pop_slice(self, min_slice_length_ms: float) -> List[SaeMessage]:
        '''
        Removes and returns all messages exceeding the target buffer length,
        if that sequence at least spans min_slice_length_ms in message time.
        If the buffer does not contain enough elements none are returned.
        '''
        if len(self._messages) == 0:
            return []
        
        cutoff_index = len(self._messages) - 1
        for idx, msg in enumerate(self._messages):
            if msg.frame.timestamp_utc_ms >= self._messages[0].frame.timestamp_utc_ms + min_slice_length_ms:
                cutoff_index = idx
                break
        if not self._messages[-1].frame.timestamp_utc_ms - self._messages[cutoff_index].frame.timestamp_utc_ms >= self._target_window_size_ms:
            return []
        
        messages = []
        while self.is_healthy():
            messages.append(self._messages.pop(0))
        return messages

    def is_healthy(self) -> bool:
        '''Check if the buffer has enough messages to satisfy the window size condition.'''
        return len(self._messages) > 0 and self._messages[-1].frame.timestamp_utc_ms - self._messages[0].frame.timestamp_utc_ms >= self._target_window_size_ms
        
    def __len__(self) -> int:
        return len(self._messages)
    
    def __getitem__(self, index) -> SaeMessage:
        return self._messages.__getitem__(index)
