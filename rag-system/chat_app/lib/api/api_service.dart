import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://localhost:8000";

  Future<Map<String, dynamic>> login(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/login'),
      body: json.encode({'username': username, 'password': password}),
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Failed to login");
    }
  }

  Future<Map<String, dynamic>> register(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/register'),
      body: json.encode({'username': username, 'password': password}),
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Failed to register");
    }
  }

  Future<List<String>> getChatHistory(int userId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/chat/history/$userId'),
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      return List<String>.from(json.decode(response.body)['history']);
    } else {
      throw Exception("Failed to load chat history");
    }
  }

  Future<Map<String, dynamic>> chat(int userId, String query, {bool newChat = false}) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      body: json.encode({'user_id': userId, 'query': query, 'new_chat': newChat}),
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Failed to chat");
    }
  }
}
