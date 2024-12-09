import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  final String baseUrl = 'http://localhost:8000'; // Update with your server URL
  String? _token;
  String? _role; // Store user role if needed

  void setToken(String token) {
    _token = token;
  }

  void setRole(String role) {
    _role = role;
  }

  String? get currentUserRole => _role;

  Map<String, String> get authHeaders {
    final headers = <String, String>{};
    if (_token != null) {
      headers['Authorization'] = 'Bearer $_token';
    }
    return headers;
  }

  /// LOGIN using form data (POST /login)
  Future<void> login(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/login'),
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: 'username=${Uri.encodeQueryComponent(username)}&password=${Uri.encodeQueryComponent(password)}',
    );

    if (response.statusCode == 200) {
      final data = json.decode(utf8.decode(response.bodyBytes));
      setToken(data['access_token']);
      // If your backend returns user role on login, set it here:
      // setRole(data['role']);
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to login');
    }
  }

  /// REGISTER using form data (POST /register)
  Future<void> register(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/register'),
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: 'username=${Uri.encodeQueryComponent(username)}&password=${Uri.encodeQueryComponent(password)}',
    );

    if (response.statusCode == 200) {
      print('Registration successful');
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to register');
    }
  }

  /// GET CHATS (GET /chats)
  Future<List<dynamic>> getChats() async {
    final response = await http.get(Uri.parse('$baseUrl/chats'), headers: authHeaders);
    if (response.statusCode == 200) {
      final data = json.decode(utf8.decode(response.bodyBytes));
      return data['chats'] ?? [];
    } else {
      throw Exception('Failed to load chats');
    }
  }

  /// GET CHAT HISTORY (GET /chat/history/{chat_id})
  Future<Map<String, dynamic>> getChatHistory(String chatId) async {
    final response = await http.get(Uri.parse('$baseUrl/chat/history/$chatId'), headers: authHeaders);
    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      throw Exception('Failed to load chat history');
    }
  }

  /// CHAT (POST /chat) - Sends a query to the assistant
  Future<Map<String, dynamic>> chat(String query, {required bool newChat, String? chatId}) async {
    final body = {'query': query, 'new_chat': newChat};
    if (chatId != null) body['chat_id'] = chatId;

    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {
        ...authHeaders,
        'Content-Type': 'application/json'
      },
      body: json.encode(body),
    );

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      throw Exception('Failed to send chat message');
    }
  }

  /// UPDATE ROLE (POST /update-role) - JSON-based
  Future<void> updateRole(String username, String newRole) async {
    final payload = {"username": username, "new_role": newRole};
    final response = await http.post(
      Uri.parse('$baseUrl/update-role'),
      headers: {
        ...authHeaders,
        'Content-Type': 'application/json'
      },
      body: json.encode(payload),
    );

    if (response.statusCode != 200) {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to update role');
    }
  }

  /// CREATE WORKSPACE (POST /workspaces) - JSON-based
  Future<Map<String, dynamic>> createWorkspace(String name) async {
    final payload = {"name": name};
    final response = await http.post(
      Uri.parse('$baseUrl/workspaces'),
      headers: {
        ...authHeaders,
        'Content-Type': 'application/json'
      },
      body: json.encode(payload),
    );

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to create workspace');
    }
  }

  /// ASSIGN USER TO WORKSPACE (POST /workspaces/{workspace_id}/assign-user) - JSON-based
  Future<Map<String, dynamic>> assignUserToWorkspace(int workspaceId, int userId) async {
    final payload = {"user_id": userId};
    final response = await http.post(
      Uri.parse('$baseUrl/workspaces/$workspaceId/assign-user'),
      headers: {
        ...authHeaders,
        'Content-Type': 'application/json'
      },
      body: json.encode(payload),
    );

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to assign user to workspace');
    }
  }

  /// UPLOAD DOCUMENT (POST /documents) - multipart/form-data
  Future<Map<String, dynamic>> uploadDocument(String filePath, String scope) async {
    var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/documents'));
    request.headers.addAll(authHeaders);
    request.files.add(await http.MultipartFile.fromPath('file', filePath));
    request.fields['scope'] = scope;

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to upload document');
    }
  }
}
