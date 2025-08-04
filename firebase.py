import firebase_admin
from firebase_admin import credentials, firestore
import os

# Check if Firebase app is already initialized
#if not firebase_admin._apps:
#    # Try to initialize with credentials file, fall back to environment variables
#    try:
#        cred_file = "lbot-e990e-firebase-adminsdk-fbsvc-f5a78c3b06.json"
#        if os.path.exists(cred_file):
#            cred = credentials.Certificate(cred_file)
#        else:
#            # Try to initialize with environment variables or default credentials
#            cred = credentials.ApplicationDefault()
#        firebase_admin.initialize_app(cred)
#    except Exception as e:
#        print(f"Warning: Firebase initialization failed: {e}")
#        print("Firebase features will not be available.")
#        # Create a dummy client for development
#        db = None
#    else:
#        db = firestore.client()
#else:
#    db = firestore.client()

#import streamlit as st
#import firebase_admin
#from firebase_admin import credentials, firestore
#
## Check if Firebase app is already initialized
#if not firebase_admin._apps:
#    try:
#        # Use credentials from Streamlit secrets
#        firebase_config = st.secrets["firebase"]
#
#        cred = credentials.Certificate({
#            "type": firebase_config["type"],
#            "project_id": firebase_config["project_id"],
#            "private_key_id": firebase_config["private_key_id"],
#            "private_key": firebase_config["private_key"].replace('\\n', '\n'),
#            "client_email": firebase_config["client_email"],
#            "client_id": firebase_config["client_id"],
#            "auth_uri": firebase_config["auth_uri"],
#            "token_uri": firebase_config["token_uri"],
#            "auth_provider_x509_cert_url": firebase_config["auth_provider_x509_cert_url"],
#            "client_x509_cert_url": firebase_config["client_x509_cert_url"],
#            "universe_domain": firebase_config["universe_domain"]
#        })
#
#        firebase_admin.initialize_app(cred)
#        db = firestore.client()
#    except Exception as e:
#        st.warning(f"Firebase initialization failed: {e}")
#        db = None
#else:
#    db = firestore.client()
#

#data = {
#    "My Payee List": "Paul, Alice, Bob",
#    "Account Balance": "$1000",
#    "Destination Accounts": "Paul, Alice, Bob, John",
#}
#
#doc_ref = db.collection("main_DB").document("user_123")
#doc_ref.set(data)


#class DatabaseManager:
#    def __init__(self, collection_name):
#        self.collection_name = collection_name
#        self.db = db
#        
#        # Ensure Firebase is available - no mock data fallback
#        if self.db is None:
#            raise RuntimeError("Firebase database connection is required. Please check your Firebase configuration.")
#
#    def add_document(self, document_id, data):
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc_ref.set(data)
#        else:
#            raise RuntimeError("Database connection not available")
#
#    def get_user_account(self, document_id):
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc = doc_ref.get()
#            if doc.exists:
#                account_name = doc.to_dict().get("Account Number", "")
#                return account_name
#            return None
#        else:
#            raise RuntimeError("Database connection not available")
#
#    def get_full_user_data(self, document_id):
#        """Get the complete user document data"""
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc = doc_ref.get()
#            if doc.exists:
#                return doc.to_dict()
#            return None
#        else:
#            raise RuntimeError("Database connection not available")
#
#    def update_document(self, document_id, data):
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc_ref.update(data)
#        else:
#            raise RuntimeError("Database connection not available")
#
#    def delete_document(self, document_id):
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc_ref.delete()
#        else:
#            raise RuntimeError("Database connection not available")
#
#    def get_payee_list(self, document_id):
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc = doc_ref.get()
#            if doc.exists:
#                data = doc.to_dict()
#                return data.get("My Payee List", "").split(", ")
#            else:
#                return []
#        else:
#            raise RuntimeError("Database connection not available")
#        
#    def get_account_balance(self, document_id):
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc = doc_ref.get()
#            if doc.exists:
#                data = doc.to_dict()
#                return data.get("Account Balance", "")
#            else:
#                return None
#        else:
#            raise RuntimeError("Database connection not available")
#        
#    def get_destination_accounts(self, document_id):
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc = doc_ref.get()
#            if doc.exists:
#                data = doc.to_dict()
#                return data.get("Destination Accounts", "").split(", ")
#            else:
#                return []
#        else:
#            raise RuntimeError("Database connection not available")
#    
#    def get_user_primary_account(self, document_id):
#        """Get the user's primary account number for transfers"""
#        if self.db:
#            doc_ref = self.db.collection(self.collection_name).document(document_id)
#            doc = doc_ref.get()
#            if doc.exists:
#                data = doc.to_dict()
#                return data.get("Account Number", None)
#            return None
#        else:
#            raise RuntimeError("Database connection not available")
#
#    def list_all_users(self):
#        """Get a list of all user document IDs in the collection"""
#        if self.db:
#            try:
#                # Get all documents in the collection
#                docs = self.db.collection(self.collection_name).stream()
#                user_ids = []
#                for doc in docs:
#                    doc_id = doc.id
#                    # Filter for user documents (assuming they start with 'user_')
#                    if doc_id.startswith('user_'):
#                        user_ids.append(doc_id)
#                return user_ids
#            except Exception as e:
#                raise RuntimeError(f"Error listing users from database: {e}")
#        else:
#            raise RuntimeError("Database connection not available")
#        
## Example usage:
#if __name__ == "__main__":
#    db_manager = DatabaseManager("main_DB")
#    
#    # Add a new document
#    #db_manager.add_document("user_456", {
#    #    "My Payee List": "Charlie, David",
#    #    "Account Balance": "$2000",
#    #    "Destination Accounts": "Charlie, David, Eve"
#    #})
#    #
#    #print("Document added successfully!")
#    
#    # Test retrieving data
#    #payee_list = db_manager.get_payee_list("user_123")
#    #balance = db_manager.get_account_balance("user_123")
#    #destinations = db_manager.get_destination_accounts("user_123")
#    
#    #print(f"Payee List: {payee_list}")
#    #print(f"Account Balance: {balance}")
#    #print(f"Destination Accounts: {destinations}")














class DatabaseManager:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self._mock_db = {
            "user_123": {
                "My Payee List": "Paul, Alice, Bob",
                "Account Balance": "$1000",
                "User Name": "Test User",
                "Account Number": "ACC123456"
            },
            "user_456": {
                "My Payee List": "Charlie, David",
                "Account Balance": "$2000",
                "User Name": "Demo User",
                "Account Number": "ACC654321"
            },
            "test_user": {
                "My Payee List": "Alice, Bob, Mike",
                "Account Balance": "$1500",
                "User Name": "Test User",
                "Account Number": "ACC999999"
            }
        }
        
    def add_document(self, document_id, data):
        self._mock_db[document_id] = data

    def get_user_account(self, document_id):
        doc = self._mock_db.get(document_id)
        if doc:
            return doc.get("Account Number", "")
        return None

    def get_full_user_data(self, document_id):
        return self._mock_db.get(document_id)

    def update_document(self, document_id, data):
        if document_id in self._mock_db:
            self._mock_db[document_id].update(data)
        else:
            raise RuntimeError("Document does not exist")

    def delete_document(self, document_id):
        if document_id in self._mock_db:
            del self._mock_db[document_id]
        else:
            raise RuntimeError("Document does not exist")

    def get_payee_list(self, document_id):
        doc = self._mock_db.get(document_id)
        if doc:
            return doc.get("My Payee List", "").split(", ")
        return []

    def get_account_balance(self, document_id):
        doc = self._mock_db.get(document_id)
        if doc:
            return doc.get("Account Balance", "")
        return None

    def get_destination_accounts(self, document_id):
        doc = self._mock_db.get(document_id)
        if doc:
            return doc.get("Destination Accounts", "").split(", ")
        return []

    def get_user_primary_account(self, document_id):
        doc = self._mock_db.get(document_id)
        if doc:
            return doc.get("Account Number", None)
        return None

    def list_all_users(self):
        return [doc_id for doc_id in self._mock_db if doc_id.startswith("user_")]


# Example usage:
if __name__ == "__main__":
    db_manager = DatabaseManager("main_DB")
    
    print(db_manager.get_payee_list("user_123"))         # ['Paul', 'Alice', 'Bob']
    print(db_manager.get_account_balance("user_123"))    # $1000
    print(db_manager.get_destination_accounts("user_123"))  # ['Paul', 'Alice', 'Bob', 'John']
    
    db_manager.add_document("user_789", {
        "My Payee List": "Eve, Frank",
        "Account Balance": "$3000",
        "Destination Accounts": "Eve, Frank, Grace",
        "Account Number": "ACC789123"
    })
    
    print(db_manager.list_all_users())  # ['user_123', 'user_456', 'user_789']
