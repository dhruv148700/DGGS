from typing import List
from .Sentence import Sentence


class Rule:
    def __init__(
        self,
        head: Sentence,
        body: List[Sentence] = None,
        name: str = None,
        tau: float = 1.0,
    ):
        if head is None:
            raise ValueError("Head must be specified.")
        self.name = name
        self.head = head
        self.body = body if body is not None else []
        self.tau = tau  # rule reliability (fixed across iterations)

    def __repr__(self):
        body_str = ", ".join(s.name for s in self.body) if self.body else "∅"
        return f"{self.name}[τ={self.tau}]: {self.head.name} ← {body_str}"
