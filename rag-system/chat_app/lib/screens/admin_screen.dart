import 'package:flutter/material.dart';
import 'package:chat_app/api/api_service.dart';
import 'package:file_picker/file_picker.dart';

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
      // Not authorized
      return Scaffold(
        appBar: AppBar(title: Text("Admin Settings"), backgroundColor: Colors.pink),
        body: Center(
          child: Text("You do not have permission to view this page."),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: Text("Admin Settings"), backgroundColor: Colors.pink),
      body: SingleChildScrollView(
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
            SizedBox(height: 20),
            if (role == "superadmin") ...[
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
              SizedBox(height: 20),
            ],
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
            SizedBox(height: 20),
            if (statusMessage.isNotEmpty)
              Text(statusMessage, style: TextStyle(fontSize: 16, color: Colors.blue)),
          ],
        ),
      ),
    );
  }
}
