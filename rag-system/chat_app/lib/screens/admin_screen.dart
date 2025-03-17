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

  // For Google Drive and storage config
  final TextEditingController gdriveFolderIdController = TextEditingController();
  String selectedDatalakeType = "local";
  bool isGlobal = false;

  // For the local datalake upload (optional workspace ID)
  final TextEditingController localDatalakeWorkspaceIdController = TextEditingController();

  // Variables
  String selectedRole = "user";
  String selectedScope = "chat";
  List<dynamic> allUsers = [];
  Map<int, List<dynamic>> userWorkspacesMap = {};
  Map<int, List<dynamic>> userChatsMap = {};

  // Moderation Policy
  final TextEditingController moderationPolicyController = TextEditingController();

  // Localized info messages
  String workspaceCardMessage = "";
  String assignUserCardMessage = "";
  String userCardMessage = "";
  String embedDocsCardMessage = "";
  String storageCardMessage = "";
  String localDatalakeCardMessage = "";
  String moderationCardMessage = "";
  String apiKeysCardMessage = "";

  // API Key Management
  final TextEditingController apiKeyNameController = TextEditingController();
  final TextEditingController apiKeyWorkspaceController = TextEditingController();
  bool apiKeyIsGlobal = true;
  List<dynamic> apiKeys = [];

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
          userCardMessage = "Error loading users: $e";
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
        userCardMessage = "Error fetching workspaces: $e";
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
        userCardMessage = "Error loading user chats: $e";
      });
    }
  }

  Future<void> assignUserToWorkspace() async {
    try {
      int workspaceId = int.parse(assignWorkspaceIdController.text.trim());
      int userId = int.parse(assignUserIdController.text.trim());
      await apiService.assignUserToWorkspace(workspaceId, userId);
      setState(() {
        assignUserCardMessage = "Assigned user $userId to workspace $workspaceId";
      });
      await fetchUserWorkspaces(userId);
    } catch (e) {
      setState(() {
        assignUserCardMessage = "Error assigning user: $e";
      });
    }
  }

  Future<void> changeUsername(int userId) async {
    final newUsername = newUsernameController.text.trim();
    if (newUsername.isEmpty) return;

    try {
      await apiService.changeUsername(userId, newUsername);
      setState(() {
        userCardMessage = "Username changed successfully for user $userId";
      });
      await loadUsersIfSuperAdmin();
      newUsernameController.clear();
    } catch (e) {
      setState(() {
        userCardMessage = "Error changing username: $e";
      });
    }
  }

  Future<void> changePassword(int userId) async {
    final newPassword = newPasswordController.text.trim();
    if (newPassword.isEmpty) return;

    try {
      await apiService.changePassword(userId, newPassword);
      setState(() {
        userCardMessage = "Password changed successfully for user $userId";
      });
      newPasswordController.clear();
    } catch (e) {
      setState(() {
        userCardMessage = "Error changing password: $e";
      });
    }
  }

  Future<void> createWorkspace() async {
    try {
      final result = await apiService.createWorkspace(workspaceNameController.text.trim());
      setState(() {
        workspaceCardMessage = "Workspace created: ${result['workspace_id']}";
      });
    } catch (e) {
      setState(() {
        workspaceCardMessage = "Error creating workspace: $e";
      });
    }
  }

  Future<void> uploadDocument() async {
    try {
      if (filePathController.text.isEmpty) {
        setState(() {
          embedDocsCardMessage = "No file selected";
        });
        return;
      }
      final response = await apiService.uploadDocument(filePathController.text, selectedScope);
      setState(() {
        embedDocsCardMessage = "Document uploaded successfully: ${response['message']}";
      });
    } catch (e) {
      setState(() {
        embedDocsCardMessage = "Error uploading document: $e";
      });
    }
  }

  Future<void> pickFile() async {
    final result = await FilePicker.platform.pickFiles();
    if (result != null) {
      filePathController.text = result.files.single.path ?? "";
    }
  }

  Future<void> embedDocuments() async {
    final directory = embedDirectoryController.text.trim();
    if (directory.isEmpty) {
      setState(() {
        embedDocsCardMessage = "Directory path cannot be empty.";
      });
      return;
    }
    try {
      await apiService.embedDocuments(directory);
      setState(() {
        embedDocsCardMessage = "Documents embedded successfully from $directory.";
      });
    } catch (e) {
      setState(() {
        embedDocsCardMessage = "Error embedding documents: $e";
      });
    }
  }

  Future<void> copyFromGDriveToLocal() async {
    final folderId = gdriveFolderIdController.text.trim();
    if (folderId.isEmpty) {
      setState(() {
        storageCardMessage = "Folder ID cannot be empty.";
      });
      return;
    }
    try {
      final result = await apiService.copyGoogleDriveToLocal(folderId, isGlobal);
      setState(() {
        storageCardMessage = result["message"];
      });
    } catch (e) {
      setState(() {
        storageCardMessage = "Error copying from Google Drive: $e";
      });
    }
  }

  Future<void> doConfigureStorage() async {
    try {
      final result = await apiService.configureStorageDashboard(selectedDatalakeType);
      setState(() {
        storageCardMessage = result["message"];
      });
    } catch (e) {
      setState(() {
        storageCardMessage = "Error configuring storage: $e";
      });
    }
  }

  Future<void> uploadFileToLocalDatalake() async {
    if (filePathController.text.isEmpty) {
      setState(() {
        localDatalakeCardMessage = "No file selected for local datalake upload.";
      });
      return;
    }

    int? workspaceId;
    if (localDatalakeWorkspaceIdController.text.trim().isNotEmpty) {
      workspaceId = int.tryParse(localDatalakeWorkspaceIdController.text.trim());
    }

    try {
      final result = await apiService.uploadFileToLocalDatalake(
        filePathController.text,
        isGlobal,
        workspaceId: workspaceId,
      );
      setState(() {
        localDatalakeCardMessage = "Local datalake upload success: ${result['message']}";
      });
    } catch (e) {
      setState(() {
        localDatalakeCardMessage = "Error uploading file to local datalake: $e";
      });
    }
  }

  Future<void> fetchModerationPolicy() async {
    try {
      final config = await apiService.getModerationConfig();
      final prompt = config['moderation_policy_prompt'] ?? '';
      moderationPolicyController.text = prompt;
      setState(() {
        moderationCardMessage = "Moderation policy loaded.";
      });
    } catch (e) {
      setState(() {
        moderationCardMessage = "Error fetching moderation policy: $e";
      });
    }
  }

  Future<void> saveModerationPolicy() async {
    try {
      final newConfig = {
        "moderation_policy_prompt": moderationPolicyController.text.trim(),
      };
      await apiService.updateModerationConfig(newConfig);
      setState(() {
        moderationCardMessage = "Moderation policy updated successfully.";
      });
    } catch (e) {
      setState(() {
        moderationCardMessage = "Error saving moderation policy: $e";
      });
    }
  }

  // API Key Management
  Future<void> generateApiKey() async {
    final name = apiKeyNameController.text.trim();
    if (name.isEmpty) {
      setState(() {
        apiKeysCardMessage = "API Key name cannot be empty.";
      });
      return;
    }

    int? wsId;
    if (apiKeyWorkspaceController.text.trim().isNotEmpty) {
      wsId = int.tryParse(apiKeyWorkspaceController.text.trim());
    }

    try {
      final result = await apiService.generateApiKey(name, apiKeyIsGlobal, workspaceId: wsId);
      setState(() {
        apiKeysCardMessage = "API Key generated: ${result['api_key_value']}";
      });
    } catch (e) {
      setState(() {
        apiKeysCardMessage = "Error generating API key: $e";
      });
    }
  }

  Future<void> listApiKeys() async {
    try {
      final keys = await apiService.listApiKeys();
      setState(() {
        apiKeys = keys;
      });
    } catch (e) {
      setState(() {
        apiKeysCardMessage = "Error listing API keys: $e";
      });
    }
  }

  /// Revoke a single API key by ID
  Future<void> revokeApiKey(int keyId) async {
    try {
      await apiService.revokeApiKey(keyId);
      setState(() {
        apiKeysCardMessage = "API Key $keyId revoked successfully.";
      });
      // Refresh the list so it no longer shows the revoked key
      await listApiKeys();
    } catch (e) {
      setState(() {
        apiKeysCardMessage = "Error revoking API key: $e";
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

            //------------------------------------------------------------------
            // Manage Workspaces Card
            //------------------------------------------------------------------
            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (workspaceCardMessage.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Text(workspaceCardMessage, style: TextStyle(color: Colors.blue)),
                      ),
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
                    if (assignUserCardMessage.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Text(assignUserCardMessage, style: TextStyle(color: Colors.blue)),
                      ),
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

            //------------------------------------------------------------------
            // User Management (superadmin)
            //------------------------------------------------------------------
            if (role == "superadmin" && allUsers.isNotEmpty)
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (userCardMessage.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Text(userCardMessage, style: TextStyle(color: Colors.blue)),
                        ),
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
                                            }
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
                                            }
                                        );
                                      },
                                      style: ElevatedButton.styleFrom(backgroundColor: Colors.orange),
                                      child: Text("Change Password"),
                                    ),
                                    SizedBox(width: 10),
                                    ElevatedButton(
                                      onPressed: () => fetchUserChats(user['id']),
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

            //------------------------------------------------------------------
            // Embed Documents
            //------------------------------------------------------------------
            if (role == "superadmin")
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (embedDocsCardMessage.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Text(embedDocsCardMessage, style: TextStyle(color: Colors.blue)),
                        ),
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
                        onPressed: embedDocuments,
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                        child: Text("Start Embedding"),
                      ),
                      SizedBox(height: 10),
                    ],
                  ),
                ),
              ),

            //------------------------------------------------------------------
            // STORAGE INTEGRATION CARD
            //------------------------------------------------------------------
            if (role == "superadmin")
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (storageCardMessage.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Text(storageCardMessage, style: TextStyle(color: Colors.blue)),
                        ),
                      Text(
                        "Storage Integration Settings",
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 10),

                      Text("Select Datalake Type:"),
                      DropdownButton<String>(
                        value: selectedDatalakeType,
                        items: <String>["local", "s3", "azureblob", "minio"].map((String value) {
                          return DropdownMenuItem<String>(
                            value: value,
                            child: Text(value),
                          );
                        }).toList(),
                        onChanged: (val) {
                          if (val != null) {
                            setState(() {
                              selectedDatalakeType = val;
                            });
                          }
                        },
                      ),
                      SizedBox(height: 10),

                      ElevatedButton(
                        onPressed: doConfigureStorage,
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                        child: Text("Configure Storage"),
                      ),

                      Divider(height: 30),

                      Text("Copy Files from Google Drive to Local"),
                      SizedBox(height: 10),
                      TextField(
                        controller: gdriveFolderIdController,
                        decoration: InputDecoration(
                          labelText: "Google Drive Folder ID",
                          border: OutlineInputBorder(),
                        ),
                      ),
                      SizedBox(height: 10),
                      Row(
                        children: [
                          Text("is_global?"),
                          Switch(
                            value: isGlobal,
                            onChanged: (val) {
                              setState(() {
                                isGlobal = val;
                              });
                            },
                          ),
                        ],
                      ),
                      SizedBox(height: 10),
                      ElevatedButton(
                        onPressed: copyFromGDriveToLocal,
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                        child: Text("Copy to Local"),
                      ),
                    ],
                  ),
                ),
              ),

            //------------------------------------------------------------------
            // Upload & Embed into Local Datalake
            //------------------------------------------------------------------
            if (role == "superadmin" || role == "admin")
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (localDatalakeCardMessage.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Text(localDatalakeCardMessage, style: TextStyle(color: Colors.blue)),
                        ),
                      Text(
                        "Upload & Embed File to Local Datalake",
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 10),
                      Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: filePathController,
                              readOnly: true,
                              decoration: InputDecoration(
                                labelText: "File Path",
                                border: OutlineInputBorder(),
                              ),
                            ),
                          ),
                          SizedBox(width: 10),
                          ElevatedButton(
                            onPressed: pickFile,
                            child: Text("Browse"),
                          ),
                        ],
                      ),
                      SizedBox(height: 10),

                      Row(
                        children: [
                          Text("is_global?"),
                          Switch(
                            value: isGlobal,
                            onChanged: (val) {
                              setState(() {
                                isGlobal = val;
                              });
                            },
                          ),
                        ],
                      ),
                      SizedBox(height: 10),

                      TextField(
                        controller: localDatalakeWorkspaceIdController,
                        decoration: InputDecoration(
                          labelText: "Workspace ID (optional)",
                          border: OutlineInputBorder(),
                        ),
                        keyboardType: TextInputType.number,
                      ),
                      SizedBox(height: 10),

                      ElevatedButton(
                        onPressed: uploadFileToLocalDatalake,
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                        child: Text("Upload & Embed"),
                      ),
                    ],
                  ),
                ),
              ),

            //------------------------------------------------------------------
            // Moderation Settings (superadmin)
            //------------------------------------------------------------------
            if (role == "superadmin")
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (moderationCardMessage.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Text(moderationCardMessage, style: TextStyle(color: Colors.blue)),
                        ),
                      Text(
                        "Moderation Settings",
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 10),
                      TextField(
                        controller: moderationPolicyController,
                        maxLines: 6,
                        decoration: InputDecoration(
                          labelText: "LLM Moderation Policy Prompt",
                          border: OutlineInputBorder(),
                        ),
                      ),
                      SizedBox(height: 10),
                      Row(
                        children: [
                          ElevatedButton(
                            onPressed: fetchModerationPolicy,
                            style: ElevatedButton.styleFrom(backgroundColor: Colors.blueGrey),
                            child: Text("Load Policy"),
                          ),
                          SizedBox(width: 10),
                          ElevatedButton(
                            onPressed: saveModerationPolicy,
                            style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                            child: Text("Save Policy"),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

            //------------------------------------------------------------------
            // API Key Management
            //------------------------------------------------------------------
            if (role == "admin" || role == "superadmin")
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (apiKeysCardMessage.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Text(apiKeysCardMessage, style: TextStyle(color: Colors.blue)),
                        ),
                      Text(
                        "API Key Management",
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 10),
                      TextField(
                        controller: apiKeyNameController,
                        decoration: InputDecoration(
                          labelText: "API Key Name",
                          border: OutlineInputBorder(),
                        ),
                      ),
                      SizedBox(height: 10),
                      Row(
                        children: [
                          Text("is_global?"),
                          Switch(
                            value: apiKeyIsGlobal,
                            onChanged: (val) {
                              setState(() {
                                apiKeyIsGlobal = val;
                              });
                            },
                          ),
                        ],
                      ),
                      SizedBox(height: 10),
                      TextField(
                        controller: apiKeyWorkspaceController,
                        decoration: InputDecoration(
                          labelText: "Workspace ID (optional)",
                          border: OutlineInputBorder(),
                        ),
                        keyboardType: TextInputType.number,
                      ),
                      SizedBox(height: 10),
                      Row(
                        children: [
                          ElevatedButton(
                            onPressed: generateApiKey,
                            style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                            child: Text("Generate API Key"),
                          ),
                          SizedBox(width: 10),
                          ElevatedButton(
                            onPressed: listApiKeys,
                            style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
                            child: Text("List API Keys"),
                          ),
                        ],
                      ),
                      SizedBox(height: 20),
                      if (apiKeys.isNotEmpty)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text("Your API Keys:", style: TextStyle(fontWeight: FontWeight.bold)),
                            for (var k in apiKeys) ...[
                              Row(
                                children: [
                                  // Display key info
                                  Expanded(
                                    child: Text(
                                      "ID: ${k['id']}, Name: ${k['name']}, is_global: ${k['is_global']}, Key: ${k['key_value']}",
                                    ),
                                  ),
                                  SizedBox(width: 10),
                                  ElevatedButton(
                                    onPressed: () => revokeApiKey(k['id']),
                                    style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                                    child: Text("Revoke"),
                                  ),
                                ],
                              ),
                              SizedBox(height: 5),
                            ],
                          ],
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
