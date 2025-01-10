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

  Future<void> login(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/login'),
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
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

  Future<void> register(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/register'),
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: 'username=${Uri.encodeQueryComponent(username)}&password=${Uri.encodeQueryComponent(password)}',
    );

    if (response.statusCode == 200) {
      print('Registration successful');
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to register');
    }
  }

  Future<List<dynamic>> getChats() async {
    final response = await http.get(Uri.parse('$baseUrl/chats'), headers: authHeaders);
    if (response.statusCode == 200) {
      final data = json.decode(utf8.decode(response.bodyBytes));
      return data['chats'] ?? [];
    } else {
      throw Exception('Failed to load chats');
    }
  }

  Future<Map<String, dynamic>> getChatHistory(String chatId) async {
    final response = await http.get(Uri.parse('$baseUrl/chat/history/$chatId'), headers: authHeaders);
    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      throw Exception('Failed to load chat history');
    }
  }

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

  // Admin & Documents

  Future<List<dynamic>> getAllUsers() async {
    final response = await http.get(Uri.parse('$baseUrl/admin/users'), headers: authHeaders);
    if (response.statusCode == 200) {
      final data = json.decode(utf8.decode(response.bodyBytes));
      return data['users'] ?? [];
    } else {
      throw Exception('Failed to load users');
    }
  }

  Future<void> changeUsername(int userId, String newUsername) async {
    final uri = Uri.parse('$baseUrl/admin/users/$userId/change-username');
    final response = await http.post(
      uri,
      headers: {
        ...authHeaders,
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: 'new_username=${Uri.encodeQueryComponent(newUsername)}',
    );
    if (response.statusCode != 200) {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to change username');
    }
  }

  Future<void> changePassword(int userId, String newPassword) async {
    final uri = Uri.parse('$baseUrl/admin/users/$userId/change-password');
    final response = await http.post(
      uri,
      headers: {
        ...authHeaders,
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: 'new_password=${Uri.encodeQueryComponent(newPassword)}',
    );
    if (response.statusCode != 200) {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to change password');
    }
  }

  Future<List<dynamic>> getUserChats(int userId) async {
    final uri = Uri.parse('$baseUrl/admin/users/$userId/chats');
    final response = await http.get(uri, headers: authHeaders);
    if (response.statusCode == 200) {
      final data = json.decode(utf8.decode(response.bodyBytes));
      return data['chats'] ?? [];
    } else {
      throw Exception('Failed to load user chats');
    }
  }

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

  Future<List<dynamic>> getUserWorkspaces(int userId) async {
    final uri = Uri.parse('$baseUrl/workspaces/$userId/list');
    final response = await http.get(uri, headers: authHeaders);

    if (response.statusCode == 200) {
      final data = json.decode(utf8.decode(response.bodyBytes));
      return data['workspaces'] ?? [];
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to load user workspaces');
    }
  }

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

  Future<Map<String, dynamic>> uploadDocument(String filePath, String scope, {String? chatId}) async {
    var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/documents'));
    request.headers.addAll(authHeaders);
    request.files.add(await http.MultipartFile.fromPath('file', filePath));
    request.fields['scope'] = scope;

    if (scope == "chat" && chatId != null) {
      request.fields['chat_id'] = chatId;
    }

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to upload document');
    }
  }

  Future<void> embedDocuments({
    required String directory,
    required bool isGlobal,
    int? workspaceId,
  }) async {
    // We'll send form-urlencoded data
    final uri = Uri.parse('$baseUrl/embed-documents');

    // Build the fields
    final Map<String, String> fields = {
      'directory': directory,
      'is_global': isGlobal.toString(),
      // Only send workspace_id if not global and we have a valid ID
    };
    if (!isGlobal && workspaceId != null) {
      fields['workspace_id'] = workspaceId.toString();
    }

    final response = await http.post(
      uri,
      headers: {
        ...authHeaders,
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: fields,
    );

    if (response.statusCode != 200) {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to embed documents');
    }
  }
}
