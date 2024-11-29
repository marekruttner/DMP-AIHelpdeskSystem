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

  @override
  void initState() {
    super.initState();
    loadChatHistory();
  }

  void loadChatHistory() async {
    try {
      final history = await apiService.getChatHistory(1); // Replace '1' with user_id
      setState(() {
        for (var convo in history) {
          messages.add({"message": convo, "isUser": convo.startsWith("User:")});
        }
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Chyba: Nelze načíst historii", "isUser": false});
      });
    }
  }

  void sendMessage({bool newChat = false}) async {
    String userMessage = messageController.text.trim();

    if (userMessage.isEmpty) return;

    setState(() {
      messages.add({"message": userMessage, "isUser": true});
    });

    messageController.clear();

    try {
      final response = await apiService.chat(1, userMessage, newChat: newChat); // Replace '1' with user_id
      String botMessage = response['response'] ?? "Chyba: Žádná odpověď";

      setState(() {
        messages.add({"message": botMessage, "isUser": false});
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Chyba: Nelze se připojit k serveru", "isUser": false});
      });
    }
  }

  void startNewChat() {
    setState(() {
      messages.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "nTech Chatbot Framework",
          style: TextStyle(color: Colors.pink, fontSize: 20),
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        actions: [
          IconButton(
            icon: Icon(Icons.chat_bubble_outline, color: Colors.blue),
            onPressed: startNewChat,
          ),
        ],
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
                      hintText: "Zadejte svou zprávu...",
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(20)),
                    ),
                  ),
                ),
                SizedBox(width: 10),
                FloatingActionButton(
                  onPressed: () => sendMessage(),
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
        ),
      ),
    );
  }
}
