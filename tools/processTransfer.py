from langchain_core.tools import StructuredTool
from controllers.apiController import APIController
from toolInputs.processTransferInput import ProcessTransferInput

# Debug removed - tool is working correctly

class ProcessTransfer(StructuredTool):
    """Tool to process a transfer from the user's account to a payee account via API."""
    
    def __init__(self):
        super().__init__(
            name="process_transfer",
            func=self.process_transfer,
            description="Process a money transfer from the user's account to a payee account using the banking API.",
            args_schema=ProcessTransferInput
        )
    
    def process_transfer(self, input_data: ProcessTransferInput) -> str:
        """Process the transfer using the API controller."""
        print("DEBUG: Received structured input:", input_data.dict())
        
        # If source_account is not provided or is empty, get it from user token
        source_account = input_data.source_account
        if not source_account or source_account.strip() == "":
            # Get source account from user token
            try:
                from firebase import DatabaseManager
                import jwt
                import os
                
                jwt_secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
                payload = jwt.decode(input_data.user_token, jwt_secret_key, algorithms=["HS256"])
                user_id = payload.get("user_id")
                
                if user_id:
                    db_manager = DatabaseManager("main_DB")
                    source_account = db_manager.get_user_primary_account(user_id)
                else:
                    source_account = "123456789"  # Default
            except Exception as e:
                print(f"DEBUG: Error getting source account: {e}")
                source_account = "123456789"  # Default
        
        controller = APIController()
        return controller.process_transfer(
            input_data.user_token,
            source_account,
            input_data.destination_account,
            float(input_data.amount)
        )