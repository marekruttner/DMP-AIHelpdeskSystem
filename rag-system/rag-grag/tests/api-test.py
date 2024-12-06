import requests
import threading

# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000"  # Update this if the server runs on a different host/port

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
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.token = None
        self.chat_id = None

    def test_register(self):
        print(f"[{self.username}] Testing /register endpoint...")
        url = f"{BASE_URL}/register"
        payload = {
            "username": self.username,
            "password": self.password
        }
        response = requests.post(url, json=payload)
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
        response = requests.post(url, json=payload)
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        response_json = response.json()
        print(f"[{self.username}] Response Body: {response_json}\n")
        if response.status_code == 200 and "access_token" in response_json:
            self.token = response_json["access_token"]  # Save the JWT token
        return response_json

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
            "query": "Jak se můžu připojit k síti eduroam?",
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
    # Step 1: Test user registration
    user.test_register()

    # Step 2: Test user login
    user.test_login()

    # Step 3: Test fetching chats (after login)
    if user.token:
        user.test_chats()

        # Step 4: Test starting a new chat
        user.test_chat(new_chat=True)

        # Step 5: Test sending a message to the existing chat
        user.test_chat(new_chat=False, chat_id=user.chat_id)

        # Step 6: Test fetching chat history
        user.test_chat_history()
    else:
        print(f"[{user.username}] Login failed. Unable to test further functionality.")

if __name__ == "__main__":
    # Create a list of TestUser instances
    users = [
        TestUser("testuser1", "testpassword1"),
        TestUser("testuser2", "testpassword2"),
        TestUser("testuser3", "testpassword3")
    ]

    threads = []
    for user in users:
        t = threading.Thread(target=user_test_scenario, args=(user,))
        t.start()
        threads.append(t)

    # Wait for all threads to complete
    for t in threads:
        t.join()
