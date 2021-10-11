class FuncInputError(Exception):
    def __init__(self, data):
        self.data = data
    
    def __str__(self) -> str:
        return self.data