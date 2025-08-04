import requests
import os
from dotenv import load_dotenv
import jwt
from firebase import DatabaseManager

load_dotenv(override=True)

class APIController():
    """Controller for handling banking API interactions using Firebase."""
    
    def __init__(self):
        self.api_key = os.getenv("CIM_API_KEY")
        self.base_url = ""
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        self.db_manager = DatabaseManager("main_DB")
    
    def verify_user_token(self, token: str) -> str:
        """Verify JWT token and return user_id."""
        try:
            # First try to decode as JWT
            payload = jwt.decode(token, self.jwt_secret_key, algorithms=["HS256"])
            return payload.get("user_id")
        except Exception:
            # For development: check if token exists as a user in database
            try:
                # Check if this token exists as a user_id in the database
                user_data = self.db_manager.get_full_user_data(token)
                if user_data:
                    # Token exists as a valid user_id in database
                    return token
                return None
            except Exception:
                return None
    
    def check_account_balance(self, user_token: str, amount: float) -> str:
        """Check if the user's primary account has sufficient balance via Firebase."""
        try:
            user_id = self.verify_user_token(user_token)
            if not user_id:
                return "Invalid or missing user token"

            # Get user data from Firebase
            user_data = self.db_manager.get_full_user_data(user_id)
            if not user_data:
                return "User account not found"
            
            # Parse balance from Firebase (format: "$1000" -> 1000)
            balance_str = user_data.get("Account Balance", "$0")
            balance = float(balance_str.replace("$", "").replace(",", ""))
            
            if amount > balance:
                return "Insufficient balance"
            return str(amount)
        
        except Exception as e:
            return f"Error checking balance: {str(e)}"
    
    def check_account_existence(self, user_token: str, account: str) -> bool:
        """Check if the account is in the user's My Payees list via Firebase."""
        try:
            user_id = self.verify_user_token(user_token)
            if not user_id:
                return False

            # Get payee list from Firebase
            payees = self.db_manager.get_payee_list(user_id)
            
            return account in payees
        
        except Exception as e:
            return False
    
    def process_transfer(self, user_token: str, source_account: str, destination_account: str, amount: float) -> str:
        """Process a transfer from source to destination account via Firebase."""
        try:
            user_id = self.verify_user_token(user_token)
            if not user_id:
                return "Invalid or missing user token"

            # Get user data from Firebase
            user_data = self.db_manager.get_full_user_data(user_id)
            if not user_data:
                return "User account not found"
            
            # Parse current balance
            balance_str = user_data.get("Account Balance", "$0")
            current_balance = float(balance_str.replace("$", "").replace(",", ""))
            
            # Check if sufficient balance
            if amount > current_balance:
                return "Transfer failed: Insufficient balance"
            
            # Calculate new balance
            new_balance = current_balance - amount
            
            # Update the balance in Firebase
            self.db_manager.update_document(user_id, {
                "Account Balance": f"${new_balance:,.2f}"
            })
            
            # Generate a mock transaction ID for response
            import time
            transaction_id = f"TXN{int(time.time())}"
            
            return f"Transfer successful: Transaction ID {transaction_id}"
        
        except Exception as e:
            return f"Error processing transfer: {str(e)}"
        

