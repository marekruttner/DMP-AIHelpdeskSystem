import 'package:flutter/material.dart';
import 'package:chat_app/api/api_service.dart';

class CommonAppBar extends StatelessWidget implements PreferredSizeWidget {
  final ApiService apiService;
  final String title;

  CommonAppBar({required this.apiService, required this.title});

  @override
  Widget build(BuildContext context) {
    return AppBar(
      title: Text(title, style: TextStyle(color: Colors.pink)),
      backgroundColor: Colors.white,
      elevation: 0,
      actions: [
        PopupMenuButton<String>(
          icon: Icon(Icons.menu, color: Colors.blue),
          onSelected: (value) {
            if (value == 'chat') {
              Navigator.pushReplacementNamed(context, '/chat');
            } else if (value == 'settings') {
              Navigator.pushReplacementNamed(context, '/settings');
            } else if (value == 'admin') {
              final role = apiService.currentUserRole;
              if (role == 'admin' || role == 'superadmin') {
                Navigator.pushReplacementNamed(context, '/admin');
              } else {
                ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('You do not have permission to access Admin Dashboard'))
                );
              }
            } else if (value == 'logout') {
              apiService.logout();
              Navigator.pushReplacementNamed(context, '/login');
            }
          },
          itemBuilder: (BuildContext context) => <PopupMenuEntry<String>>[
            const PopupMenuItem<String>(
              value: 'chat',
              child: Text('Chat'),
            ),
            const PopupMenuItem<String>(
              value: 'settings',
              child: Text('Settings'),
            ),
            const PopupMenuItem<String>(
              value: 'admin',
              child: Text('Admin Dashboard'),
            ),
            const PopupMenuItem<String>(
              value: 'logout',
              child: Text('Logout'),
            ),
          ],
        )
      ],
    );
  }

  @override
  Size get preferredSize => Size.fromHeight(kToolbarHeight);
}
