from .Sentence import Sentence


class Assumption(Sentence):
    def __init__(
        self,
        name: str,
        contrary: str = None,
        tau: float = 0.5,
    ):
        super().__init__(name)
        self.contrary = contrary  # name of the contrary sentence
        self.tau = tau            # base strength (fixed across iterations)

    def __eq__(self, other):
        if not isinstance(other, Assumption):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Assumption({self.name}, contrary={self.contrary}, tau={self.tau:.4f})"
