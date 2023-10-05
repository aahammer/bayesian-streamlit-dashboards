from pydantic import BaseModel, confloat

class BetaPrior(BaseModel):
    """
    Represents the prior knowledge about a beta distribution.
    """
    alpha: confloat(gt=0)
    beta: confloat(gt=0)

