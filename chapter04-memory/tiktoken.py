class _Encoding:
    def __init__(self, name: str):
        self.name = name

    def encode(self, text: str):
        return [ord(ch) for ch in text]


def get_encoding(name: str):
    return _Encoding(name)


def encoding_for_model(name: str):
    return _Encoding(name)

