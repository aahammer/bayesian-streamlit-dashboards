from pydantic import BaseModel, confloat

class BetaPrior(BaseModel):
    """
    Represents the prior knowledge about a beta distribution.
    """
    alpha: confloat(ge=0)
    beta: confloat(ge=0)

