import requests
import threading

# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000"

# Helper function to set headers with JWT token
def get_headers(token):
    if token:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    else:
        return {"Content-Type": "application/json"}

class TestUser:
    def __init__(self, username, password, role="user"):
        self.username = username
        self.password = password
        self.role = role
        self.token = None
        self.chat_id = None
        self.workspace_id = None

    def test_register(self):
        print(f"[{self.username}] Testing /register endpoint...")
        url = f"{BASE_URL}/register"
        payload = {
            "username": self.username,
            "password": self.password
        }
        response = requests.post(url, data=payload)  # Changed from json=payload to data=payload
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        print(f"[{self.username}] Response Body: {response.json()}\n")
        return response.json()

    def test_login(self):
        print(f"[{self.username}] Testing /login endpoint...")
        url = f"{BASE_URL}/login"
        payload = {
            "username": self.username,
            "password": self.password
        }
        response = requests.post(url, data=payload)  # Changed from json=payload to data=payload
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        response_json = response.json()
        print(f"[{self.username}] Response Body: {response_json}\n")
        if response.status_code == 200 and "access_token" in response_json:
            self.token = response_json["access_token"]  # Save the JWT token
        return response_json

    def test_create_workspace(self):
        if self.role not in ["admin", "superadmin"]:
            print(f"[{self.username}] Insufficient permissions to create a workspace.\n")
            return
        print(f"[{self.username}] Testing /workspaces endpoint...")
        url = f"{BASE_URL}/workspaces"
        payload = {"name": f"{self.username}_workspace"}
        response = requests.post(url, json=payload, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        print(f"[{self.username}] Response Body: {response.json()}\n")
        if response.status_code == 200:
            self.workspace_id = response.json().get("workspace_id")
        return response.json()

    def test_assign_user_to_workspace(self, user_id):
        if self.role not in ["admin", "superadmin"]:
            print(f"[{self.username}] Insufficient permissions to assign a user to a workspace.\n")
            return
        print(f"[{self.username}] Testing /workspaces/{self.workspace_id}/assign-user endpoint...")
        url = f"{BASE_URL}/workspaces/{self.workspace_id}/assign-user"
        payload = {"user_id": user_id}
        response = requests.post(url, json=payload, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        print(f"[{self.username}] Response Body: {response.json()}\n")
        return response.json()

    def test_upload_document(self, scope):
        print(f"[{self.username}] Testing /documents endpoint with scope {scope}...")
        url = f"{BASE_URL}/documents"
        files = {'file': ('test_doc.txt', 'This is a test document for scope testing.')}
        data = {'scope': scope}
        response = requests.post(url, files=files, data=data, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        print(f"[{self.username}] Response Body: {response.json()}\n")
        return response.json()

    def test_chats(self):
        print(f"[{self.username}] Testing /chats endpoint...")
        url = f"{BASE_URL}/chats"
        response = requests.get(url, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Response Body: {response.json()}\n")
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
        return response.json()

    def test_chat(self, new_chat=False, chat_id=None):
        print(f"[{self.username}] Testing /chat endpoint (new_chat={new_chat})...")
        url = f"{BASE_URL}/chat"
        payload = {
            "query": "How do I connect to the network?",
            "new_chat": new_chat
        }
        if chat_id:
            payload["chat_id"] = chat_id
        response = requests.post(url, json=payload, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Response Body: {response.json()}\n")
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
        response_json = response.json()
        if "chat_id" in response_json and new_chat:
            self.chat_id = response_json["chat_id"]
        return response_json

    def test_chat_history(self):
        if not self.chat_id:
            print(f"[{self.username}] No chat_id available to fetch history.\n")
            return
        print(f"[{self.username}] Testing /chat/history endpoint...")
        url = f"{BASE_URL}/chat/history/{self.chat_id}"
        response = requests.get(url, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Response Body: {response.json()}\n")
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
        return response.json()

def user_test_scenario(user):
    user.test_register()
    user.test_login()
    if user.token:
        if user.role in ["admin", "superadmin"]:
            user.test_create_workspace()
            user.test_assign_user_to_workspace(user_id=2)  # Assign another user
        user.test_upload_document(scope="chat")
        if user.role in ["admin", "superadmin"]:
            user.test_upload_document(scope="workspace")
        if user.role == "superadmin":
            user.test_upload_document(scope="system")
        user.test_chats()
        user.test_chat(new_chat=True)
        user.test_chat(new_chat=False, chat_id=user.chat_id)
        user.test_chat_history()
    else:
        print(f"[{user.username}] Login failed. Unable to test further functionality.")

if __name__ == "__main__":
    users = [
        TestUser("user1", "password1", "user"),
        TestUser("admin1", "password1", "admin"),
        TestUser("superadmin1", "password1", "superadmin")
    ]

    threads = []
    for user in users:
        t = threading.Thread(target=user_test_scenario, args=(user,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
