import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Singleton implementation
  static final ApiService _instance = ApiService._internal();

  factory ApiService() {
    return _instance;
  }

  ApiService._internal();

  final String baseUrl = 'http://localhost:8000'; // Replace with your server's base URL
  String? _token; // JWT token storage

  // Set the token after login
  void setToken(String token) {
    _token = token;
  }

  // Headers with token
  Map<String, String> get headers => {
    'Content-Type': 'application/json',
    if (_token != null) 'Authorization': 'Bearer $_token',
  };

  // Login function to fetch and store token
  Future<void> login(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/login'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'username': username, 'password': password}),
    );

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      setToken(data['access_token']); // Save the token here
      print('Token saved: $_token'); // Debug token
    } else {
      final errorData = json.decode(response.body);
      throw Exception(errorData['detail'] ?? 'Failed to login');
    }
  }

  // Registration function
  Future<void> register(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/register'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'username': username, 'password': password}),
    );

    if (response.statusCode == 200) {
      print('Registration successful'); // Debug success
    } else {
      final errorData = json.decode(response.body);
      throw Exception(errorData['detail'] ?? 'Failed to register');
    }
  }

  // Fetch chats
  Future<List<dynamic>> getChats() async {
    print('Headers in getChats: $headers'); // Debug headers
    final response = await http.get(Uri.parse('$baseUrl/chats'), headers: headers);
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return data['chats'] ?? [];
    } else {
      throw Exception('Failed to load chats');
    }
  }

  // Fetch chat history
  Future<Map<String, dynamic>> getChatHistory(String chatId) async {
    print('Headers in getChatHistory: $headers'); // Debug headers
    final response = await http.get(Uri.parse('$baseUrl/chat/history/$chatId'), headers: headers);
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to load chat history');
    }
  }

  // Send chat message
  Future<Map<String, dynamic>> chat(String query, {required bool newChat, String? chatId}) async {
    final body = {'query': query, 'new_chat': newChat};
    if (chatId != null) body['chat_id'] = chatId;

    print('Headers in chat: $headers'); // Debug headers
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: headers,
      body: json.encode(body),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to send chat message');
    }
  }
}
