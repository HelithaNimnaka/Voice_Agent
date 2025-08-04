#from pydantic import BaseModel
#
#class CheckAccountBalanceInput(BaseModel):
#    amount: int
#    user_token: str
#
    

from pydantic import BaseModel, Field
from typing import Union

class CheckAccountBalanceInput(BaseModel):
    amount: Union[int, float] = Field(
        description="The amount to check if user has sufficient balance for (e.g., 400)"
    )
    user_token: str = Field(
        description="The user's authentication token, typically 'user123'"
    )
