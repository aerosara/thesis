import pandas as pd

def _custom__getattr__(self, key):
    # If attribute is in the self Series instance ...
    if key in self:
        # ... return is as an attribute
        return self[key]
    else:
        # ... raise the usual exception
        raise AttributeError("'Series' object has no attribute '%s'" % key)

# Overwrite current Series attributes 'else' case
pd.Series.__getattr__ = _custom__getattr__
