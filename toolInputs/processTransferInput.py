from pydantic import BaseModel
from typing import Optional

class ProcessTransferInput(BaseModel):
    source_account: Optional[str] = ""
    destination_account: str
    amount: int
    user_token: str