import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:chat_app/api/api_service.dart';

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController messageController = TextEditingController();
  final List<Map<String, dynamic>> messages = [];
  final ApiService apiService = ApiService();
  final ScrollController _scrollController = ScrollController(); // Scroll controller for chat
  List<Map<String, dynamic>> chats = [];
  String? currentChatId;
  String? currentChatName;

  @override
  void initState() {
    super.initState();
    fetchAllChats(); // Fetch all existing chats when the screen loads
  }

  Future<void> fetchAllChats() async {
    try {
      final chatList = await apiService.getChats();
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

      final history = await apiService.getChatHistory(chatId);
      print("Chat history received: ${history['history']}");

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

        print("Parsed messages: $messages");
      });

      _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
    } catch (e) {
      print("Error loading chat history: $e");
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

      _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
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
                onTap: () {
                  Navigator.pop(context);
                  loadChatHistory(chat['chatId'], chat['name']);
                },
              );
            }).toList(),
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
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              key: ValueKey(messages.length),
              itemCount: messages.length,
              itemBuilder: (context, index) {
                final chat = messages[index];
                return ChatBubble(
                  key: ValueKey(index),
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

  const ChatBubble({Key? key, required this.message, required this.isUser}) : super(key: key);

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
            ? Text(
          message, // Plain text for user messages
          style: TextStyle(fontSize: 16, height: 1.5),
          textDirection: TextDirection.ltr,
        )
            : MarkdownBody(
          data: message, // Render AI responses as Markdown
          styleSheet: MarkdownStyleSheet(
            p: TextStyle(fontSize: 16, height: 1.5),
            listBullet: TextStyle(fontSize: 16, height: 1.5),
          ),
          onTapLink: (text, href, title) {
            if (href != null) {
              // Handle link taps (e.g., open in a browser)
              print("Tapped link: $href");
            }
          },
        ),
      ),
    );
  }
}

