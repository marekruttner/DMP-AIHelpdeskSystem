import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  final String baseUrl = 'http://100.85.225.62:8000';
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
      body:
      'username=${Uri.encodeQueryComponent(username)}&password=${Uri.encodeQueryComponent(password)}',
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
      body:
      'username=${Uri.encodeQueryComponent(username)}&password=${Uri.encodeQueryComponent(password)}',
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
    final response = await http.get(
      Uri.parse('$baseUrl/chat/history/$chatId'),
      headers: authHeaders,
    );
    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      throw Exception('Failed to load chat history');
    }
  }

  Future<Map<String, dynamic>> chat(String query,
      {required bool newChat, String? chatId}) async {
    final body = {'query': query, 'new_chat': newChat};
    if (chatId != null) body['chat_id'] = chatId;

    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {
        ...authHeaders,
        'Content-Type': 'application/json',
      },
      body: json.encode(body),
    );

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      throw Exception('Failed to send chat message');
    }
  }

  // -------------------------------------------------------------------------
  // Rate an AI response
  // -------------------------------------------------------------------------
  Future<void> rateResponse(String chatId, int rating, {String? comment}) async {
    final uri = Uri.parse('$baseUrl/rate_response');
    final payload = {
      "chat_id": chatId,
      "rating": rating,
    };
    if (comment != null && comment.isNotEmpty) {
      payload["comment"] = comment;
    }

    final response = await http.post(
      uri,
      headers: {
        ...authHeaders,
        'Content-Type': 'application/json',
      },
      body: json.encode(payload),
    );

    if (response.statusCode != 200) {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to rate response');
    }
  }

  // -------------------------------------------------------------------------
  // Admin & Documents
  // -------------------------------------------------------------------------
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
        'Content-Type': 'application/x-www-form-urlencoded',
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
        'Content-Type': 'application/x-www-form-urlencoded',
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
        'Content-Type': 'application/json',
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
        'Content-Type': 'application/json',
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
        'Content-Type': 'application/json',
      },
      body: json.encode(payload),
    );

    if (response.statusCode != 200) {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to update role');
    }
  }

  Future<Map<String, dynamic>> uploadDocument(String filePath, String scope, {String? chatId}) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/documents'),
    );
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

  Future<void> embedDocuments(String directory) async {
    final response = await http.post(
      Uri.parse('$baseUrl/embed-documents'),
      headers: {
        ...authHeaders,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: 'directory=${Uri.encodeQueryComponent(directory)}',
    );

    if (response.statusCode != 200) {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to embed documents');
    }
  }

  // -------------------------------------------------------------------------
  // Copy from GDrive -> local, configure storage, etc.
  // -------------------------------------------------------------------------
  Future<Map<String, dynamic>> copyGoogleDriveToLocal(String folderId, bool isGlobal) async {
    final uri = Uri.parse('$baseUrl/admin/copy-google-drive-to-local');
    final request = http.MultipartRequest('POST', uri);
    request.headers.addAll(authHeaders);

    request.fields['folder_id'] = folderId;
    request.fields['is_global'] = isGlobal.toString();

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to copy from Google Drive to local.');
    }
  }

  Future<Map<String, dynamic>> configureStorageDashboard(String datalakeType) async {
    final uri = Uri.parse('$baseUrl/admin/configure-storage-dashboard');
    final request = http.MultipartRequest('POST', uri);
    request.headers.addAll(authHeaders);

    request.fields['datalake_type'] = datalakeType;

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to configure storage');
    }
  }

  Future<Map<String, dynamic>> uploadFileToLocalDatalake(
      String filePath,
      bool isGlobal, {
        int? workspaceId,
      }) async {
    final uri = Uri.parse('$baseUrl/local-datalake/upload-file');
    final request = http.MultipartRequest('POST', uri);

    request.headers.addAll(authHeaders);

    request.files.add(await http.MultipartFile.fromPath('file', filePath));

    request.fields['is_global'] = isGlobal.toString();
    if (workspaceId != null) {
      request.fields['workspace_id'] = workspaceId.toString();
    }

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes));
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to upload file to local datalake.');
    }
  }

  // -------------------------------------------------------------------------
  // LLM Moderation Config
  // -------------------------------------------------------------------------
  Future<Map<String, dynamic>> getModerationConfig() async {
    final response = await http.get(
      Uri.parse('$baseUrl/admin/moderation-config'),
      headers: authHeaders,
    );
    if (response.statusCode == 200) {
      return jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to fetch moderation config');
    }
  }

  Future<Map<String, dynamic>> updateModerationConfig(Map<String, dynamic> newConfig) async {
    final response = await http.post(
      Uri.parse('$baseUrl/admin/moderation-config'),
      headers: {
        ...authHeaders,
        'Content-Type': 'application/json',
      },
      body: jsonEncode(newConfig),
    );
    if (response.statusCode == 200) {
      return jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to update moderation config');
    }
  }

  // -------------------------------------------------------------------------
  // API Key Management
  // -------------------------------------------------------------------------
  Future<Map<String, dynamic>> generateApiKey(String name, bool isGlobal, {int? workspaceId}) async {
    final uri = Uri.parse('$baseUrl/admin/api-keys/generate');
    final payload = {
      "name": name,
      "is_global": isGlobal,
    };
    if (workspaceId != null) {
      payload["workspace_id"] = workspaceId;
    }

    final response = await http.post(
      uri,
      headers: {
        ...authHeaders,
        'Content-Type': 'application/json',
      },
      body: jsonEncode(payload),
    );

    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to generate API key');
    }
  }

  Future<List<dynamic>> listApiKeys() async {
    final uri = Uri.parse('$baseUrl/admin/api-keys');
    final response = await http.get(uri, headers: authHeaders);
    if (response.statusCode == 200) {
      final data = json.decode(utf8.decode(response.bodyBytes));
      return data['api_keys'] ?? [];
    } else {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to list API keys');
    }
  }

  /// Revoke (delete) an API key, given its integer `apiKeyId`.
  /// Adjust the endpoint path if your server uses a different route.
  Future<void> revokeApiKey(int apiKeyId) async {
    final uri = Uri.parse('$baseUrl/admin/api-keys/$apiKeyId');
    final response = await http.delete(
      uri,
      headers: authHeaders,
    );

    if (response.statusCode != 200 && response.statusCode != 204) {
      final errorData = json.decode(utf8.decode(response.bodyBytes));
      throw Exception(errorData['detail'] ?? 'Failed to revoke API key');
    }
  }
}
