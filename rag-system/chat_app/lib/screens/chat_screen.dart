import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:file_picker/file_picker.dart';
import 'package:chat_app/api/api_service.dart';
import 'package:chat_app/widgets/common_app_bar.dart';

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final ApiService apiService = ApiService();

  // The messages displayed in the chat
  final List<Map<String, dynamic>> messages = [];

  // Chat & UI state
  List<Map<String, dynamic>> chats = [];
  String? currentChatId;
  String? currentChatName;
  bool isTyping = false;

  // SINGLE-workspace UI selection (the old popup menu) => null => "All"
  int? selectedWorkspaceId;

  // MULTI-workspace selection data
  // Each item: { "workspace_id": int, "name": String, "selected": bool }
  List<Map<String, dynamic>> userWorkspaces = [];

  // ExcludeGlobal switch
  bool excludeGlobal = false;

  @override
  void initState() {
    super.initState();
    fetchAllChats();
    fetchUserWorkspaces();
  }

  // 1) Load existing user chats
  Future<void> fetchAllChats() async {
    try {
      final chatList = await apiService.getChats();
      setState(() {
        chats = chatList.map<Map<String, dynamic>>((chat) => {
          "chatId": chat['chat_id'],
          // Hide chat content from user => no "latestMessage"
          "name": "Chat ${chat['chat_id'].substring(0, 6)}",
          "latestMessage": chat['latest_message'] ?? ""
        }).toList();
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to load chats", "isUser": false});
      });
    }
  }

  // 2) Load user’s available workspaces
  Future<void> fetchUserWorkspaces() async {
    try {
      final userId = await apiService.getCurrentUserId();
      final wsList = await apiService.getUserWorkspaces(userId);
      setState(() {
        userWorkspaces.clear();
        for (var ws in wsList) {
          userWorkspaces.add({
            "workspace_id": ws["workspace_id"],
            "name": ws["name"],
            "selected": false,
          });
        }
      });
    } catch (e) {
      print("Error fetching user workspaces: $e");
    }
  }

  // 3) Load chat history
  void loadChatHistory(String chatId, String chatName) async {
    try {
      currentChatId = chatId;
      currentChatName = chatName;

      final history = await apiService.getChatHistory(chatId);
      setState(() {
        messages.clear();

        String currentSpeaker = "";
        StringBuffer currentMessage = StringBuffer();

        for (var convo in history['history']) {
          if (convo.startsWith("User:")) {
            if (currentMessage.isNotEmpty) {
              messages.add({
                "message": currentMessage.toString().trim(),
                "isUser": currentSpeaker == "User",
              });
              currentMessage.clear();
            }
            currentSpeaker = "User";
            currentMessage.write(convo.replaceFirst("User:", "").trim());
          } else if (convo.startsWith("AI:")) {
            if (currentMessage.isNotEmpty) {
              messages.add({
                "message": currentMessage.toString().trim(),
                "isUser": currentSpeaker == "User",
              });
              currentMessage.clear();
            }
            currentSpeaker = "AI";
            currentMessage.write(convo.replaceFirst("AI:", "").trim());
          } else {
            currentMessage.write("\n$convo");
          }
        }

        if (currentMessage.isNotEmpty) {
          messages.add({
            "message": currentMessage.toString().trim(),
            "isUser": currentSpeaker == "User",
          });
        }
      });

      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scrollController.hasClients) {
          _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
        }
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to load chat history", "isUser": false});
      });
    }
  }

  // 4) Send user message
  Future<void> sendMessage() async {
    final userMessage = messageController.text.trim();
    if (userMessage.isEmpty) return;

    setState(() {
      messages.add({"message": userMessage, "isUser": true});
      isTyping = true;
    });
    messageController.clear();

    try {
      bool newChat = currentChatId == null;

      // MULTI selection
      List<int> multiSelected = [];
      for (var w in userWorkspaces) {
        if (w["selected"] == true) {
          multiSelected.add(w["workspace_id"]);
        }
      }

      // SINGLE selection
      if (selectedWorkspaceId != null && !multiSelected.contains(selectedWorkspaceId)) {
        multiSelected.add(selectedWorkspaceId!);
      }
      final workspaceIds = multiSelected.isEmpty ? null : multiSelected;

      // pass excludeGlobal
      final response = await apiService.chat(
        userMessage,
        newChat: newChat,
        chatId: currentChatId,
        workspaceIds: workspaceIds,
        excludeGlobal: excludeGlobal,  // ensure we pass this
      );

      final botMessage = response['response'] ?? "No response received";

      if (newChat) {
        currentChatId = response['chat_id'];
        currentChatName = "New Chat";
        await fetchAllChats();
      }

      setState(() {
        messages.add({"message": botMessage, "isUser": false});
        isTyping = false;
      });

      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scrollController.hasClients) {
          _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
        }
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to send message: $e", "isUser": false});
        isTyping = false;
      });
    }
  }

  // 5) Embed a document
  Future<void> embedDocument() async {
    if (currentChatId == null) {
      setState(() {
        messages.add({
          "message": "Cannot embed a document in a new chat. Please send a message first.",
          "isUser": false
        });
      });
      return;
    }

    try {
      final result = await FilePicker.platform.pickFiles();
      if (result == null) return; // user canceled

      final filePath = result.files.single.path;
      if (filePath == null) return;

      await apiService.uploadDocument(filePath, "chat", chatId: currentChatId!);
      setState(() {
        messages.add({"message": "Document embedded successfully.", "isUser": false});
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to embed document: $e", "isUser": false});
      });
    }
  }

  void startNewChat() {
    setState(() {
      messages.clear();
      currentChatId = null;
      currentChatName = null;
    });
  }

  // Show multi-workspace bottom sheet
  void showWorkspaceSelection() {
    showModalBottomSheet(
        context: context,
        builder: (BuildContext context) {
          return ListView(
            children: [
              ListTile(
                title: Text("Select Workspaces"),
                trailing: IconButton(
                  icon: Icon(Icons.check),
                  onPressed: () {
                    Navigator.pop(context);
                    setState(() {});
                  },
                ),
              ),
              Divider(),
              for (var ws in userWorkspaces)
                CheckboxListTile(
                  title: Text(ws["name"]),
                  value: ws["selected"],
                  onChanged: (bool? value) {
                    setState(() {
                      ws["selected"] = value ?? false;
                    });
                  },
                ),
            ],
          );
        }
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(currentChatName ?? "Select a Chat"),
        actions: [
          // Toggle "excludeGlobal"
          Row(
            children: [
              Text("Exclude Global", style: TextStyle(fontSize: 14)),
              Switch(
                value: excludeGlobal,
                onChanged: (val) {
                  setState(() {
                    excludeGlobal = val;
                  });
                },
              ),
            ],
          ),

          // SINGLE-workspace "All or One" popup menu
          if (userWorkspaces.isNotEmpty)
            PopupMenuButton<int?>(
              child: Row(
                children: [
                  Text(
                    selectedWorkspaceId == null
                        ? "All Workspaces"
                        : userWorkspaces.firstWhere(
                            (ws) => ws["workspace_id"] == selectedWorkspaceId,
                        orElse: () => {"name": "Unknown"}
                    )["name"],
                    style: TextStyle(color: Colors.white),
                  ),
                  Icon(Icons.arrow_drop_down, color: Colors.white),
                ],
              ),
              onSelected: (int? value) {
                setState(() {
                  selectedWorkspaceId = value;
                });
              },
              itemBuilder: (context) {
                return [
                  PopupMenuItem<int?>(
                    value: null,
                    child: Text("All Workspaces"),
                  ),
                  ...userWorkspaces.map((ws) {
                    return PopupMenuItem<int?>(
                      value: ws["workspace_id"],
                      child: Text(ws["name"]),
                    );
                  }).toList(),
                ];
              },
            ),

          // The multi-workspace selection icon
          IconButton(
            icon: Icon(Icons.group_work),
            onPressed: showWorkspaceSelection,
          ),
        ],
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(
              decoration: BoxDecoration(color: Colors.pink),
              child: Text(
                "Chats",
                style: TextStyle(color: Colors.white, fontSize: 24),
              ),
            ),
            // Hide chat content => no subtitle
            for (var chat in chats)
              ListTile(
                title: Text(chat['name']),
                onTap: () {
                  Navigator.pop(context);
                  loadChatHistory(chat['chatId'], chat['name']);
                },
              ),
            ListTile(
              leading: Icon(Icons.add),
              title: Text("Start New Chat"),
              onTap: () {
                Navigator.pop(context);
                startNewChat();
              },
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          // The chat messages
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: messages.length + (isTyping ? 1 : 0),
              itemBuilder: (context, index) {
                if (isTyping && index == messages.length) {
                  return TypingIndicator();
                }
                final chat = messages[index];
                return ChatBubble(
                  message: chat['message'],
                  isUser: chat['isUser'],
                );
              },
            ),
          ),

          // Input row
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                // embed button
                IconButton(
                  icon: Icon(Icons.add, color: Colors.pink),
                  onPressed: embedDocument,
                ),
                // message field
                Expanded(
                  child: TextField(
                    controller: messageController,
                    decoration: InputDecoration(
                      hintText: "Type your message...",
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(20)),
                    ),
                  ),
                ),
                SizedBox(width: 10),
                // send
                FloatingActionButton(
                  onPressed: sendMessage,
                  backgroundColor: Colors.pink,
                  child: Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// A bubble widget for chat messages
class ChatBubble extends StatelessWidget {
  final String message;
  final bool isUser;
  const ChatBubble({Key? key, required this.message, required this.isUser})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 5, horizontal: 10),
        padding: EdgeInsets.all(15),
        decoration: BoxDecoration(
          color: isUser ? Colors.pink[100] : Colors.grey[200],
          borderRadius: BorderRadius.circular(15),
        ),
        child: isUser
            ? Text(message, style: TextStyle(fontSize: 16, height: 1.5))
            : MarkdownBody(
          data: message,
          styleSheet: MarkdownStyleSheet(
            p: TextStyle(fontSize: 16, height: 1.5),
            listBullet: TextStyle(fontSize: 16, height: 1.5),
          ),
        ),
      ),
    );
  }
}

// A typing indicator widget
class TypingIndicator extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 5, horizontal: 10),
        padding: EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: Colors.grey[200],
          borderRadius: BorderRadius.circular(15),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            DotAnimation(),
            SizedBox(width: 5),
            Text("AI is typing...", style: TextStyle(fontSize: 14, color: Colors.grey[700])),
          ],
        ),
      ),
    );
  }
}

// The "three dots" animation
class DotAnimation extends StatefulWidget {
  @override
  _DotAnimationState createState() => _DotAnimationState();
}

class _DotAnimationState extends State<DotAnimation>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<int> _dotCountAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: Duration(seconds: 1),
      vsync: this,
    )..repeat();

    _dotCountAnimation = StepTween(begin: 1, end: 3).animate(_controller);
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _dotCountAnimation,
      builder: (context, child) {
        String dots = "." * _dotCountAnimation.value;
        return Text(dots, style: TextStyle(fontSize: 18));
      },
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
