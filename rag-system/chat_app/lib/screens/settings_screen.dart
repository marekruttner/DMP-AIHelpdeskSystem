import 'package:flutter/material.dart';
import 'package:chat_app/api/api_service.dart';

class SettingsScreen extends StatelessWidget {
  final ApiService apiService = ApiService();

  @override
  Widget build(BuildContext context) {
    final role = apiService.currentUserRole;
    return Scaffold(
      appBar: AppBar(
        title: Text("Settings"),
        backgroundColor: Colors.pink,
      ),
      body: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          children: [
            Text("User Settings", style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
            SizedBox(height: 20),
            Text("Username: ...", style: TextStyle(fontSize: 18)),
            Text("Role: $role", style: TextStyle(fontSize: 18)),
            SizedBox(height: 20),
            if (role == "admin" || role == "superadmin")
              ElevatedButton(
                onPressed: () {
                  Navigator.pushNamed(context, '/admin');
                },
                style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                child: Text("Admin Settings"),
              ),
          ],
        ),
      ),
    );
  }
}
