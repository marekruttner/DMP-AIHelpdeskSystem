import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  final String baseUrl = 'http://localhost:8000';
  String? _token;
  String? _role;

  void setToken(String token) {
    _token = token.isEmpty ? null : token;
  }

  void setRole(String role) {
    _role = role.isEmpty ? null : role;
  }

  String? get currentUserRole => _role;

  /// Clears the stored token and role, effectively logging the user out.
  void logout() {
    _token = null;
    _role = null;
  }

  Map<String, String> get authHeaders {
    final headers = <String, String>{};
    if (_token != null) {
      headers['Authorization'] = 'Bearer $_token';
    }
    return headers;
  }

  /// Logs the user in by sending form data (username, password) to /login
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
      if (data.containsKey('role')) {
        setRole(data['role']);
      }
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to login');
    }
  }

  /// Registers a new user by sending form data to /register
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

  /// Fetches a list of chats available to the user
  Future<List<dynamic>> getChats() async {
    final response = await http.get(Uri.parse('$baseUrl/chats'), headers: authHeaders);
    if (response.statusCode == 200) {
      final data = json.decode(utf8.decode(response.bodyBytes));
      return data['chats'] ?? [];
    } else {
      throw Exception('Failed to load chats');
    }
  }

  /// Fetches the chat history for a given chat_id
  Future<Map<String, dynamic>> getChatHistory(String chatId) async {
    final response = await http.get(Uri.parse('$baseUrl/chat/history/$chatId'), headers: authHeaders);
    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      throw Exception('Failed to load chat history');
    }
  }

  /// Sends a message to the chat endpoint, creating a new chat if newChat = true
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

  /// ADMIN & DOCUMENT METHODS EXAMPLES:

  /// Update a user's role (superadmin or admin only)
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

  /// Create a new workspace (admin or superadmin only)
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

  /// Assign a user to a workspace (admin or superadmin only)
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

  /// Upload a document (user, admin, or superadmin)
  /// scope can be: chat, profile, workspace, system
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
