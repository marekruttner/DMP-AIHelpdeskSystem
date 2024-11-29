import requests

# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000"  # Update this if the server runs on a different host/port

# Global token variable to store the JWT for authenticated requests
token = None

# Helper function to set headers with JWT token
def get_headers():
    global token
    if token:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    else:
        return {"Content-Type": "application/json"}

def test_register():
    print("Testing /register endpoint...")
    url = f"{BASE_URL}/register"
    payload = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = requests.post(url, json=payload)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}\n")
    return response.json()

def test_login():
    global token
    print("Testing /login endpoint...")
    url = f"{BASE_URL}/login"
    payload = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = requests.post(url, json=payload)
    print(f"Response Status Code: {response.status_code}")
    response_json = response.json()
    print(f"Response Body: {response_json}\n")
    if response.status_code == 200 and "access_token" in response_json:
        token = response_json["access_token"]  # Save the JWT token globally
    return response_json

def test_chats():
    print("Testing /chats endpoint...")
    url = f"{BASE_URL}/chats"
    response = requests.get(url, headers=get_headers())
    print(f"Response Status Code: {response.status_code}")
    try:
        print(f"Response Body: {response.json()}\n")
    except requests.exceptions.JSONDecodeError:
        print(f"Non-JSON Response Body: {response.text}\n")
    return response.json()

def test_chat(new_chat=False, chat_id=None):
    print(f"Testing /chat endpoint (new_chat={new_chat})...")
    url = f"{BASE_URL}/chat"
    payload = {
        "query": "Jak se můžu připojit k síti eduroam?",
        "new_chat": new_chat
    }
    if chat_id:
        payload["chat_id"] = chat_id
    response = requests.post(url, json=payload, headers=get_headers())
    print(f"Response Status Code: {response.status_code}")
    try:
        print(f"Response Body: {response.json()}\n")
    except requests.exceptions.JSONDecodeError:
        print(f"Non-JSON Response Body: {response.text}\n")
    return response.json()

def test_chat_history(chat_id):
    print("Testing /chat/history endpoint...")
    url = f"{BASE_URL}/chat/history/{chat_id}"
    response = requests.get(url, headers=get_headers())
    print(f"Response Status Code: {response.status_code}")
    try:
        print(f"Response Body: {response.json()}\n")
    except requests.exceptions.JSONDecodeError:
        print(f"Non-JSON Response Body: {response.text}\n")
    return response.json()

if __name__ == "__main__":
    # Step 1: Test user registration
    register_response = test_register()

    # Step 2: Test user login
    login_response = test_login()

    # Step 3: Test fetching chats (after login)
    if token:
        chats_response = test_chats()

        # Step 4: Test starting a new chat
        new_chat_response = test_chat(new_chat=True)
        if "chat_id" in new_chat_response:
            chat_id = new_chat_response["chat_id"]

            # Step 5: Test sending a message to the existing chat
            test_chat(new_chat=False, chat_id=chat_id)

            # Step 6: Test fetching chat history
            test_chat_history(chat_id)
    else:
        print("Login failed. Unable to test further functionality.")
