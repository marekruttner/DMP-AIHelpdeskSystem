import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:chat_app/api/api_service.dart';
import 'package:file_picker/file_picker.dart';
import 'package:chat_app/widgets/common_app_bar.dart';

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController messageController = TextEditingController();
  final List<Map<String, dynamic>> messages = [];
  final ApiService apiService = ApiService();
  final ScrollController _scrollController = ScrollController();
  List<Map<String, dynamic>> chats = [];
  String? currentChatId;
  String? currentChatName;
  bool isTyping = false; // Track typing status

  @override
  void initState() {
    super.initState();
    fetchAllChats();
  }

  Future<void> fetchAllChats() async {
    try {
      final chatList = await apiService.getChats();
      setState(() {
        chats = chatList.map<Map<String, dynamic>>((chat) => {
          "chatId": chat['chat_id'],
          "name": "Chat ${chat['chat_id'].substring(0, 6)}",
          "latestMessage": chat['latest_message'] ?? "No messages yet"
        }).toList();
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

      _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
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
      isTyping = true; // Show typing indicator
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
        isTyping = false; // Hide typing indicator
      });

      _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to send message", "isUser": false});
        isTyping = false; // Hide typing indicator on error
      });
    }
  }

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
      FilePickerResult? result = await FilePicker.platform.pickFiles();
      if (result == null) return;

      final filePath = result.files.single.path;
      if (filePath == null) return;

      final response = await apiService.uploadDocument(filePath, "chat", chatId: currentChatId!);
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CommonAppBar(apiService: apiService, title: currentChatName ?? "Select a Chat"),
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
              itemCount: messages.length + (isTyping ? 1 : 0),
              itemBuilder: (context, index) {
                if (isTyping && index == messages.length) {
                  return TypingIndicator(); // Show typing indicator at the end
                }

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
                IconButton(
                  icon: Icon(Icons.add, color: Colors.pink),
                  onPressed: embedDocument,
                ),
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

class DotAnimation extends StatefulWidget {
  @override
  _DotAnimationState createState() => _DotAnimationState();
}

class _DotAnimationState extends State<DotAnimation> with SingleTickerProviderStateMixin {
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
