class Sentence:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Sentence):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Sentence({self.name})"
