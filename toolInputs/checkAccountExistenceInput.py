#from pydantic import BaseModel
#
#class CheckAccountExistenceInput(BaseModel):
#    account: str
#    user_token: str


from pydantic import BaseModel, Field
from typing import Union

class CheckAccountExistenceInput(BaseModel):
    account: Union[str, dict] = Field(
        description="The name of the account/payee to check (e.g., 'Paul', 'Alice', 'Bob')"
    )
    user_token: str = Field(
        description="The user's authentication token, typically 'user_123'"
    )
