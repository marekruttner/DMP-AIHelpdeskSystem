import 'package:flutter/material.dart';
import 'package:chat_app/api/api_service.dart';
import 'package:file_picker/file_picker.dart';
import 'package:chat_app/widgets/common_app_bar.dart';

class AdminScreen extends StatefulWidget {
  @override
  _AdminScreenState createState() => _AdminScreenState();
}

class _AdminScreenState extends State<AdminScreen> {
  final ApiService apiService = ApiService();

  final TextEditingController workspaceNameController = TextEditingController();
  final TextEditingController assignUserIdController = TextEditingController();
  final TextEditingController assignWorkspaceIdController = TextEditingController();
  final TextEditingController roleUsernameController = TextEditingController();
  String selectedRole = "user";

  final TextEditingController filePathController = TextEditingController();
  String selectedScope = "chat";

  String statusMessage = "";

  // New controller for the embed documents directory
  final TextEditingController embedDirectoryController = TextEditingController();

  // User management variables
  List<dynamic> allUsers = [];
  Map<int, List<dynamic>> userChatsMap = {}; // cache user chats
  final TextEditingController newUsernameController = TextEditingController();
  final TextEditingController newPasswordController = TextEditingController();

  @override
  void initState() {
    super.initState();
    loadUsersIfSuperAdmin();
  }

  Future<void> loadUsersIfSuperAdmin() async {
    final role = apiService.currentUserRole;
    if (role == "superadmin") {
      try {
        final users = await apiService.getAllUsers();
        setState(() {
          allUsers = users;
        });
      } catch (e) {
        setState(() {
          statusMessage = "Error loading users: $e";
        });
      }
    }
  }

  Future<void> changeUsername(int userId) async {
    final newUsername = newUsernameController.text.trim();
    if (newUsername.isEmpty) return;

    try {
      await apiService.changeUsername(userId, newUsername);
      setState(() {
        statusMessage = "Username changed successfully for user $userId";
      });
      await loadUsersIfSuperAdmin();
      newUsernameController.clear();
    } catch (e) {
      setState(() {
        statusMessage = "Error changing username: $e";
      });
    }
  }

  Future<void> changePassword(int userId) async {
    final newPassword = newPasswordController.text.trim();
    if (newPassword.isEmpty) return;

    try {
      await apiService.changePassword(userId, newPassword);
      setState(() {
        statusMessage = "Password changed successfully for user $userId";
      });
      newPasswordController.clear();
    } catch (e) {
      setState(() {
        statusMessage = "Error changing password: $e";
      });
    }
  }

  Future<void> viewUserChats(int userId) async {
    try {
      final chats = await apiService.getUserChats(userId);
      setState(() {
        userChatsMap[userId] = chats;
      });
    } catch (e) {
      setState(() {
        statusMessage = "Error loading user chats: $e";
      });
    }
  }

  Future<void> createWorkspace() async {
    try {
      final result = await apiService.createWorkspace(workspaceNameController.text.trim());
      setState(() {
        statusMessage = "Workspace created: ${result['workspace_id']}";
      });
    } catch (e) {
      setState(() {
        statusMessage = "Error creating workspace: $e";
      });
    }
  }

  Future<void> assignUserToWorkspace() async {
    try {
      int workspaceId = int.parse(assignWorkspaceIdController.text.trim());
      int userId = int.parse(assignUserIdController.text.trim());
      final result = await apiService.assignUserToWorkspace(workspaceId, userId);
      setState(() {
        statusMessage = "Assigned user $userId to workspace $workspaceId";
      });
    } catch (e) {
      setState(() {
        statusMessage = "Error assigning user: $e";
      });
    }
  }

  Future<void> updateUserRole() async {
    try {
      await apiService.updateRole(roleUsernameController.text.trim(), selectedRole);
      setState(() {
        statusMessage = "User role updated successfully";
      });
    } catch (e) {
      setState(() {
        statusMessage = "Error updating role: $e";
      });
    }
  }

  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null) {
      filePathController.text = result.files.single.path ?? "";
    }
  }

  Future<void> uploadDocument() async {
    try {
      if (filePathController.text.isEmpty) {
        setState(() {
          statusMessage = "No file selected";
        });
        return;
      }
      final response = await apiService.uploadDocument(filePathController.text, selectedScope);
      setState(() {
        statusMessage = "Document uploaded successfully: ${response['message']}";
      });
    } catch (e) {
      setState(() {
        statusMessage = "Error uploading document: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final role = apiService.currentUserRole;
    if (role != "admin" && role != "superadmin") {
      return Scaffold(
        appBar: CommonAppBar(apiService: apiService, title: "Admin Settings"),
        body: Center(
          child: Text("You do not have permission to view this page."),
        ),
      );
    }

    return Scaffold(
      appBar: CommonAppBar(apiService: apiService, title: "Admin Settings"),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (statusMessage.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 20),
                child: Text(statusMessage, style: TextStyle(fontSize: 16, color: Colors.blue)),
              ),

            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("Manage Workspaces", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                    SizedBox(height: 10),
                    TextField(
                      controller: workspaceNameController,
                      decoration: InputDecoration(
                        labelText: "Workspace Name",
                        border: OutlineInputBorder(),
                      ),
                    ),
                    SizedBox(height: 10),
                    ElevatedButton(
                      onPressed: createWorkspace,
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                      child: Text("Create Workspace"),
                    ),
                    SizedBox(height: 20),
                    Text("Assign User to Workspace", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                    SizedBox(height: 10),
                    TextField(
                      controller: assignWorkspaceIdController,
                      decoration: InputDecoration(
                        labelText: "Workspace ID",
                        border: OutlineInputBorder(),
                      ),
                      keyboardType: TextInputType.number,
                    ),
                    SizedBox(height: 10),
                    TextField(
                      controller: assignUserIdController,
                      decoration: InputDecoration(
                        labelText: "User ID",
                        border: OutlineInputBorder(),
                      ),
                      keyboardType: TextInputType.number,
                    ),
                    SizedBox(height: 10),
                    ElevatedButton(
                      onPressed: assignUserToWorkspace,
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                      child: Text("Assign User"),
                    ),
                  ],
                ),
              ),
            ),

            SizedBox(height: 20),

            if (role == "superadmin")
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Update User Role", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                      SizedBox(height: 10),
                      TextField(
                        controller: roleUsernameController,
                        decoration: InputDecoration(
                          labelText: "Username",
                          border: OutlineInputBorder(),
                        ),
                      ),
                      SizedBox(height: 10),
                      DropdownButton<String>(
                        value: selectedRole,
                        items: ["user", "admin", "superadmin"].map((r) {
                          return DropdownMenuItem(value: r, child: Text(r));
                        }).toList(),
                        onChanged: (val) {
                          if (val != null) {
                            setState(() {
                              selectedRole = val;
                            });
                          }
                        },
                      ),
                      ElevatedButton(
                        onPressed: updateUserRole,
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                        child: Text("Update Role"),
                      ),
                    ],
                  ),
                ),
              ),

            SizedBox(height: 20),

// Show Embed Documents card only if user is superadmin
            if (role == "superadmin")
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Embed Documents", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                      SizedBox(height: 10),
                      TextField(
                        controller: embedDirectoryController,
                        decoration: InputDecoration(
                          labelText: "Directory Path",
                          border: OutlineInputBorder(),
                        ),
                      ),
                      SizedBox(height: 10),
                      ElevatedButton(
                        onPressed: () async {
                          final directory = embedDirectoryController.text.trim();
                          if (directory.isEmpty) {
                            setState(() {
                              statusMessage = "Directory path cannot be empty.";
                            });
                            return;
                          }
                          try {
                            await apiService.embedDocuments(directory);
                            setState(() {
                              statusMessage = "Documents embedded successfully from $directory.";
                            });
                          } catch (e) {
                            setState(() {
                              statusMessage = "Error embedding documents: $e";
                            });
                          }
                        },
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                        child: Text("Start Embedding"),
                      ),
                    ],
                  ),
                ),
              ),

            SizedBox(height: 20),

            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("Upload Document", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                    SizedBox(height: 10),
                    TextField(
                      controller: filePathController,
                      decoration: InputDecoration(
                        labelText: "File Path",
                        border: OutlineInputBorder(),
                      ),
                      readOnly: true,
                    ),
                    SizedBox(height: 10),
                    ElevatedButton(
                      onPressed: pickFile,
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                      child: Text("Pick File"),
                    ),
                    SizedBox(height: 10),
                    Text("Scope:"),
                    DropdownButton<String>(
                      value: selectedScope,
                      items: ["chat", "profile", "workspace", "system"].map((scope) {
                        return DropdownMenuItem(value: scope, child: Text(scope));
                      }).toList(),
                      onChanged: (val) {
                        if (val != null) {
                          setState(() {
                            selectedScope = val;
                          });
                        }
                      },
                    ),
                    SizedBox(height: 10),
                    ElevatedButton(
                      onPressed: uploadDocument,
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                      child: Text("Upload Document"),
                    ),
                  ],
                ),
              ),
            ),

            SizedBox(height: 20),

            if (role == "superadmin" && allUsers.isNotEmpty)
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("All Users", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                      SizedBox(height: 10),
                      for (var user in allUsers) ...[
                        Card(
                          elevation: 2,
                          child: Padding(
                            padding: EdgeInsets.all(8),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text("ID: ${user['id']}, Username: ${user['username']}, Role: ${user['role']}"),
                                SizedBox(height: 5),
                                Row(
                                  children: [
                                    ElevatedButton(
                                      onPressed: () {
                                        // Show dialog to change username
                                        showDialog(
                                            context: context,
                                            builder: (context) {
                                              return AlertDialog(
                                                title: Text("Change Username"),
                                                content: TextField(
                                                  controller: newUsernameController,
                                                  decoration: InputDecoration(labelText: "New Username"),
                                                ),
                                                actions: [
                                                  TextButton(onPressed: () => Navigator.pop(context), child: Text("Cancel")),
                                                  ElevatedButton(
                                                    onPressed: () {
                                                      Navigator.pop(context);
                                                      changeUsername(user['id']);
                                                    },
                                                    child: Text("Change"),
                                                  )
                                                ],
                                              );
                                            }
                                        );
                                      },
                                      style: ElevatedButton.styleFrom(backgroundColor: Colors.blue),
                                      child: Text("Change Username"),
                                    ),
                                    SizedBox(width: 10),
                                    ElevatedButton(
                                      onPressed: () {
                                        // Show dialog to change password
                                        showDialog(
                                            context: context,
                                            builder: (context) {
                                              return AlertDialog(
                                                title: Text("Change Password"),
                                                content: TextField(
                                                  controller: newPasswordController,
                                                  decoration: InputDecoration(labelText: "New Password"),
                                                  obscureText: true,
                                                ),
                                                actions: [
                                                  TextButton(onPressed: () => Navigator.pop(context), child: Text("Cancel")),
                                                  ElevatedButton(
                                                    onPressed: () {
                                                      Navigator.pop(context);
                                                      changePassword(user['id']);
                                                    },
                                                    child: Text("Change"),
                                                  )
                                                ],
                                              );
                                            }
                                        );
                                      },
                                      style: ElevatedButton.styleFrom(backgroundColor: Colors.orange),
                                      child: Text("Change Password"),
                                    ),
                                    SizedBox(width: 10),
                                    ElevatedButton(
                                      onPressed: () {
                                        viewUserChats(user['id']);
                                      },
                                      style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
                                      child: Text("View Chats"),
                                    ),
                                  ],
                                ),
                                if (userChatsMap.containsKey(user['id']))
                                  Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      SizedBox(height: 10),
                                      Text("User Chats:", style: TextStyle(fontWeight: FontWeight.bold)),
                                      for (var c in userChatsMap[user['id']]!) ...[
                                        Text("Chat ID: ${c['chat_id']}, Latest Msg: ${c['latest_message'] ?? 'No messages yet'}"),
                                      ]
                                    ],
                                  )
                              ],
                            ),
                          ),
                        ),
                        SizedBox(height: 10),
                      ]
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
