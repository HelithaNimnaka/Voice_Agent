from langchain_core.tools import StructuredTool
from toolInputs.checkAccountBalanceInput import CheckAccountBalanceInput
from controllers.apiController import APIController

class CheckAccountBalance(StructuredTool):
    """Tool to check if the user's primary account has sufficient balance via API."""
    
    def __init__(self):
        super().__init__(
            name="check_account_balance",
            func=self.check_balance,
            description="Check if the user's primary account has sufficient balance for the transaction via the banking API.",
            args_schema=CheckAccountBalanceInput
        )
    
    def check_balance(self, input_data: CheckAccountBalanceInput) -> str:
        """Check if the user's account balance is sufficient using the API controller."""
        # Extract amount and user_token from the input object
        amount = input_data.amount
        user_token = input_data.user_token
        
        controller = APIController()
        return controller.check_account_balance(user_token, float(amount))