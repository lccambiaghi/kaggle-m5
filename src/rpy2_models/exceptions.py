class ModelStructureNotFoundError(Exception):
    def __init__(self, msg='Failed to find a model structure given the training data'):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class ModelNotFitError(Exception):
    def __init__(self, msg='Failed to fit the model given the training data'):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class CrossValidationScoreError(Exception):
    def __init__(self, msg='Failed to compute the cross validation score given the training data'):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)
