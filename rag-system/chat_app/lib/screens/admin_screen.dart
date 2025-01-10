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

  // Controllers
  final TextEditingController workspaceNameController = TextEditingController();
  final TextEditingController assignUserIdController = TextEditingController();
  final TextEditingController assignWorkspaceIdController = TextEditingController();
  final TextEditingController roleUsernameController = TextEditingController();
  final TextEditingController filePathController = TextEditingController();
  final TextEditingController embedDirectoryController = TextEditingController();
  final TextEditingController newUsernameController = TextEditingController();
  final TextEditingController newPasswordController = TextEditingController();
  // New controller for specifying workspace ID when embedding (if not global)
  final TextEditingController embedWorkspaceIdController = TextEditingController();

  // Variables
  String selectedRole = "user";
  String selectedScope = "chat";
  String statusMessage = "";
  List<dynamic> allUsers = [];
  Map<int, List<dynamic>> userWorkspacesMap = {};
  Map<int, List<dynamic>> userChatsMap = {};

  // New variable to toggle Global vs. Workspace
  bool isGlobalSelected = true;

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

  Future<void> fetchUserWorkspaces(int userId) async {
    try {
      final workspaces = await apiService.getUserWorkspaces(userId);
      setState(() {
        userWorkspacesMap[userId] = workspaces;
      });
    } catch (e) {
      setState(() {
        statusMessage = "Error fetching workspaces: $e";
      });
    }
  }

  Future<void> fetchUserChats(int userId) async {
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

  Future<void> assignUserToWorkspace() async {
    try {
      int workspaceId = int.parse(assignWorkspaceIdController.text.trim());
      int userId = int.parse(assignUserIdController.text.trim());
      await apiService.assignUserToWorkspace(workspaceId, userId);
      setState(() {
        statusMessage = "Assigned user $userId to workspace $workspaceId";
      });
      await fetchUserWorkspaces(userId);
    } catch (e) {
      setState(() {
        statusMessage = "Error assigning user: $e";
      });
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

  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null) {
      filePathController.text = result.files.single.path ?? "";
    }
  }

  Future<void> embedDocuments() async {
    final directory = embedDirectoryController.text.trim();
    if (directory.isEmpty) {
      setState(() {
        statusMessage = "Directory path cannot be empty.";
      });
      return;
    }

    try {
      int? workspaceId;
      if (!isGlobalSelected && embedWorkspaceIdController.text.isNotEmpty) {
        workspaceId = int.parse(embedWorkspaceIdController.text.trim());
      }

      // Call the updated apiService method
      await apiService.embedDocuments(
        directory: directory,
        isGlobal: isGlobalSelected,
        workspaceId: workspaceId,
      );

      setState(() {
        statusMessage = "Documents embedded successfully from $directory.";
      });
    } catch (e) {
      setState(() {
        statusMessage = "Error embedding documents: $e";
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
                child: Text(
                  statusMessage,
                  style: TextStyle(fontSize: 16, color: Colors.blue),
                ),
              ),
            // Workspace Management
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
            // User Management
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
                                                TextButton(
                                                  onPressed: () => Navigator.pop(context),
                                                  child: Text("Cancel"),
                                                ),
                                                ElevatedButton(
                                                  onPressed: () {
                                                    Navigator.pop(context);
                                                    changeUsername(user['id']);
                                                  },
                                                  child: Text("Change"),
                                                )
                                              ],
                                            );
                                          },
                                        );
                                      },
                                      style: ElevatedButton.styleFrom(backgroundColor: Colors.blue),
                                      child: Text("Change Username"),
                                    ),
                                    SizedBox(width: 10),
                                    ElevatedButton(
                                      onPressed: () {
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
                                                TextButton(
                                                  onPressed: () => Navigator.pop(context),
                                                  child: Text("Cancel"),
                                                ),
                                                ElevatedButton(
                                                  onPressed: () {
                                                    Navigator.pop(context);
                                                    changePassword(user['id']);
                                                  },
                                                  child: Text("Change"),
                                                )
                                              ],
                                            );
                                          },
                                        );
                                      },
                                      style: ElevatedButton.styleFrom(backgroundColor: Colors.orange),
                                      child: Text("Change Password"),
                                    ),
                                    SizedBox(width: 10),
                                    ElevatedButton(
                                      onPressed: () {
                                        fetchUserChats(user['id']);
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
            // Embed Documents
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

                      // Toggle for Global vs Workspace
                      Row(
                        children: [
                          Text("Global? "),
                          Switch(
                            value: isGlobalSelected,
                            onChanged: (newValue) {
                              setState(() {
                                isGlobalSelected = newValue;
                                if (isGlobalSelected) {
                                  embedWorkspaceIdController.clear();
                                }
                              });
                            },
                          ),
                        ],
                      ),
                      if (!isGlobalSelected) ...[
                        TextField(
                          controller: embedWorkspaceIdController,
                          keyboardType: TextInputType.number,
                          decoration: InputDecoration(
                            labelText: "Workspace ID",
                            border: OutlineInputBorder(),
                          ),
                        ),
                        SizedBox(height: 10),
                      ],

                      ElevatedButton(
                        onPressed: embedDocuments,
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                        child: Text("Start Embedding"),
                      ),
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
