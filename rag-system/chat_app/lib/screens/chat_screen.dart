import 'package:flutter/material.dart';
import 'package:chat_app/api/api_service.dart';

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController messageController = TextEditingController();
  final List<Map<String, dynamic>> messages = [];
  final ApiService apiService = ApiService();
  List<Map<String, dynamic>> chats = []; // List to store all chats with IDs
  String? currentChatId; // Currently selected chat ID
  String? currentChatName; // Currently selected chat name for display

  @override
  void initState() {
    super.initState();
    fetchAllChats(); // Fetch all existing chats when the screen loads
  }

  Future<void> fetchAllChats() async {
    try {
      final chatList = await apiService.getChats(); // No user_id required
      setState(() {
        chats = chatList
            .map<Map<String, dynamic>>((chat) => {
          "chatId": chat['chat_id'],
          "name": "Chat ${chat['chat_id'].substring(0, 6)}",
          "latestMessage": chat['latest_message'] ?? "No messages yet"
        })
            .toList();
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to load chats", "isUser": false});
      });
    }
  }

  void loadChatHistory(String chatId, String chatName) async {
    try {
      currentChatId = chatId;
      currentChatName = chatName;

      final history = await apiService.getChatHistory(chatId); // Fetch chat history
      setState(() {
        messages.clear();
        for (var convo in history['history']) {
          if (convo.startsWith("User:")) {
            messages.add({
              "message": convo.replaceFirst("User:", "").trim(),
              "isUser": true,
            });
          } else if (convo.startsWith("AI:")) {
            messages.add({
              "message": convo.replaceFirst("AI:", "").trim(),
              "isUser": false,
            });
          }
        }
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to load chat history", "isUser": false});
      });
    }
  }


  Future<void> sendMessage() async {
    String userMessage = messageController.text.trim();

    if (userMessage.isEmpty) return;

    setState(() {
      messages.add({"message": userMessage, "isUser": true});
    });

    messageController.clear();

    try {
      bool newChat = currentChatId == null;

      final response = await apiService.chat(
        userMessage,
        newChat: newChat,
        chatId: currentChatId,
      );
      String botMessage = response['response'] ?? "No response received";

      if (newChat) {
        currentChatId = response['chat_id'];
        currentChatName = "New Chat";
        await fetchAllChats();
      }

      setState(() {
        messages.add({"message": botMessage, "isUser": false});
      });
    } catch (e) {
      print("Error: $e");
      setState(() {
        messages.add({"message": "Failed to send message", "isUser": false});
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          currentChatName ?? "Select a Chat",
          style: TextStyle(color: Colors.pink, fontSize: 20),
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        actions: [
          IconButton(
            icon: Icon(Icons.add, color: Colors.blue),
            onPressed: () {
              startNewChat();
            },
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
            ...chats.map((chat) {
              return ListTile(
                title: Text(chat['name']),
                //subtitle: Text(chat['latestMessage']),
                onTap: () {
                  Navigator.pop(context); // Close the drawer
                  loadChatHistory(chat['chatId'], chat['name']); // Load selected chat
                },
              );
            }).toList(),
            ListTile(
              leading: Icon(Icons.add),
              title: Text("Start New Chat"),
              onTap: () {
                Navigator.pop(context); // Close the drawer
                startNewChat();
              },
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: messages.length,
              itemBuilder: (context, index) {
                final chat = messages[index];
                return ChatBubble(
                  message: chat['message'],
                  isUser: chat['isUser'],
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
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

class ChatBubble extends StatelessWidget {
  final String message;
  final bool isUser;

  const ChatBubble({required this.message, required this.isUser});

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
        child: Text(
          message,
          style: TextStyle(fontSize: 16, height: 1.5),
          textDirection: TextDirection.ltr, // Ensures proper text rendering
        ),
      ),
    );
  }
}
