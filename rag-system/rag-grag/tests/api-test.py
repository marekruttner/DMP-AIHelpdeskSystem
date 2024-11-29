import requests

# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000"  # Update this if the server runs on a different host/port

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
    print("Testing /login endpoint...")
    url = f"{BASE_URL}/login"
    payload = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = requests.post(url, json=payload)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}\n")
    return response.json()

def test_chat(user_id):
    print("Testing /chat endpoint...")
    url = f"{BASE_URL}/chat"
    payload = {
        "user_id": user_id,
        "query": "Jak se můžu připojit k síti eduroam?"
    }
    response = requests.post(url, json=payload)
    print(f"Response Status Code: {response.status_code}")
    try:
        print(f"Response Body: {response.json()}\n")
    except requests.exceptions.JSONDecodeError:
        print(f"Non-JSON Response Body: {response.text}\n")
    return response


if __name__ == "__main__":
    # Step 1: Test user registration
    register_response = test_register()

    # Step 2: Test user login
    login_response = test_login()
    if "user_id" in login_response:
        user_id = login_response["user_id"]

        # Step 3: Test chat functionality
        chat_response = test_chat(user_id)
    else:
        print("Login failed. Unable to test chat functionality.")
