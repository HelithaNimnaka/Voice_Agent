from langchain_core.tools import StructuredTool
from toolInputs.checkAccountExistenceInput import CheckAccountExistenceInput
from controllers.apiController import APIController

class CheckAccountExistence(StructuredTool):
    """Tool to check if an account is in the user's My Payees list via API."""
    
    def __init__(self):
        super().__init__(
            name="check_account_existence",
            func=self.check_existence,
            description="Check if the specified account is in the user's My Payees list using the banking API.",
            args_schema=CheckAccountExistenceInput
        )
    
    #def check_existence(self, account: str, user_token: str) -> str:
    #    """Check if the account exists in the user's payee list using the API controller."""
    #    controller = APIController()
    #    return controller.check_account_existence(user_token, account)
    

    def check_existence(self, input_data: CheckAccountExistenceInput) -> str:
        """Check if the account exists in the user's payee list using the API controller."""
        # Extract account and user_token from the input object
        account = input_data.account  # Changed back from payee_name
        user_token = input_data.user_token
        
        # Fix if account is an object instead of string
        if isinstance(account, dict):
            account = account.get("name", "UNKNOWN")
        elif not isinstance(account, str):
            account = str(account)
            
        # Debug: Show which user and account we're checking
        print(f"🔍 DEBUG: Checking account '{account}' for user '{user_token}'")
        
        controller = APIController()
        exists = controller.check_account_existence(user_token, account)
        
        # Debug: Show the result
        print(f"🔍 DEBUG: Account existence result: {exists}")
        
        if exists:
            return f"Account '{account}' is in your payees list."
        else:
            return f"Account '{account}' is not found in your payees list."
